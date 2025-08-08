"""Neural network to predict zernike coefficients from donut images and positions."""
from .utils import batched_crop, get_centers, convert_zernikes_deploy, single_conv
import torch
from torch import nn
from .lightning_wavenet import WaveNetSystem
from .lightning_alignnet import AlignNetSystem
from .aggregatornet import AggregatorNet
import yaml
from torch import vmap
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F_loss
from lsst.obs.lsst import LsstCam
from lsst.obs.lsst.cameraTransforms import LsstCameraTransforms
import torchvision.transforms.functional as F
from lsst.ip.isr import AssembleCcdTask
from lsst.meas.algorithms import subtractBackground
import copy


class NeuralActiveOpticsSys(pl.LightningModule):
    """Transfer learning driven WaveNet."""
    def __init__(self, dataset_params, wavenet_path=None, alignet_path=None,
                 aggregatornet_path=None,
                 lr=1e-3, final_layer=None, aggregator_on=True) -> None:
        """Initialize the Neural Active Optics System.

        Parameters
        ----------
        dataset_params : str
            Path to YAML file containing dataset parameters and configuration.
        wavenet_path : str, optional
            Path to pre-trained WaveNet model checkpoint. If None, creates new model.
        alignet_path : str, optional
            Path to pre-trained AlignNet model checkpoint. If None, creates new model.
        aggregatornet_path : str, optional
            Path to pre-trained AggregatorNet model checkpoint. If None, creates new model.
        lr : float, optional
            Learning rate for optimization. Defaults to 1e-3.
        final_layer : int, optional
            Size of additional final linear layer. If None, uses identity function.
        aggregator_on : bool, optional
            Whether to use aggregator network for final prediction. Defaults to True.
        """
        super(NeuralActiveOpticsSys, self).__init__()
        self.save_hyperparameters()
        self.device_val = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load parameters from YAML file once
        with open(dataset_params, 'r') as yaml_file:
            params = yaml.safe_load(yaml_file)

        if wavenet_path is None:
            self.wavenet_model = WaveNetSystem().to(self.device_val)
        else:
            self.wavenet_model = WaveNetSystem.load_from_checkpoint(wavenet_path).to(self.device_val)

        if alignet_path is None:
            self.alignnet_model = AlignNetSystem().to(self.device_val)
        else:
            try:
                self.alignnet_model = AlignNetSystem.load_from_checkpoint(alignet_path).to(self.device_val)
                print("✅ Loaded AlignNet regular checkpoint")
            except Exception:
                print("⚠️  Regular AlignNet loading failed")
                print("🔄 Trying to load AlignNet as QAT-trained model...")
                from training.load_qat_model import load_qat_trained_model
                self.alignnet_model = load_qat_trained_model(alignet_path, device=str(self.device_val)).to(self.device_val)
                print("✅ Loaded AlignNet QAT-trained model")

        self.max_seq_length = params["max_seq_len"]
        self.aggregator_on = aggregator_on
        print(self.device_val)
        if aggregatornet_path is None:
            d_model = params["aggregator_model"]["d_model"]
            nhead = params["aggregator_model"]["nhead"]
            num_layers = params["aggregator_model"]["num_layers"]
            dim_feedforward = params["aggregator_model"]["dim_feedforward"]
            self.aggregatornet_model = AggregatorNet(d_model=d_model,
                                                     nhead=nhead,
                                                     num_layers=num_layers,
                                                     dim_feedforward=dim_feedforward,
                                                     max_seq_length=self.max_seq_length).to(self.device_val)
        else:
            self.aggregatornet_model = AggregatorNet.load_from_checkpoint(aggregatornet_path).to(self.device_val)

        if final_layer is not None:
            layers = [
                nn.ReLU(),
                nn.Linear(19, final_layer),
            ]
            self.final_layer = nn.Sequential(*layers)
        else:
            self.final_layer = self.identity

        # Use already loaded params
        self.refinements = params['refinements']
        self.CROP_SIZE = params['CROP_SIZE']
        self.mm_pix = params['mm_pix']
        self.deg_per_pix = params['deg_per_pix']
        self.alpha = params['alpha']
        self.SCALE = params['adjustment_AlignNet']

    def identity(self, x):
        """Return the input unchanged (identity function).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            The same input tensor unchanged.
        """
        return x

    def single_conv_batched(self, data):
        """Apply single convolution operation to batched data using vectorized mapping.

        Parameters
        ----------
        data : torch.Tensor
            Batched input tensor for convolution operation.

        Returns
        -------
        torch.Tensor
            Result of applying single convolution to each element in the batch.
        """
        # Get device from the data tensor to ensure compatibility with quantized models
        device = data.device
        return vmap(lambda x: single_conv(x, device=str(device)))(data)

    def convert_zernike_device(self, data):
        """Convert Zernike coefficients format for device computation.

        Parameters
        ----------
        data : torch.Tensor
            Input Zernike coefficients tensor.

        Returns
        -------
        torch.Tensor
            Converted Zernike coefficients in proper format.
        """
        # Use the device from the data tensor to ensure compatibility
        device_str = 'cpu' if data.device.type == 'cpu' else 'cuda'
        return convert_zernikes_deploy(data, device=device_str)

    def convert_zernike_batched(self, data):
        """Apply Zernike conversion to batched data using vectorized mapping.

        Parameters
        ----------
        data : torch.Tensor
            Batched Zernike coefficients tensor.

        Returns
        -------
        torch.Tensor
            Batch of converted Zernike coefficients.
        """
        return vmap(self.convert_zernike_device)(data)

    def forward_align(
        self,
        image: torch.Tensor,
        fx: torch.Tensor,
        fy: torch.Tensor,
        intra: torch.Tensor,
        band: torch.Tensor,
    ) -> torch.Tensor:
        """Perform alignment and cropping of donut images for detection purposes.

        This method runs the alignment pipeline to center and crop donut images
        from full field exposures, applying iterative refinement using AlignNet
        and filtering based on signal-to-noise ratio criteria.

        Parameters
        ----------
        image : torch.Tensor
            Input donut image tensor of shape (batch_size, channels, height, width).
        fx : torch.Tensor
            Field x-coordinates tensor.
        fy : torch.Tensor
            Field y-coordinates tensor.
        intra : torch.Tensor
            Intra/extra-focal indicator tensor (0 for intra, 1 for extra).
        band : torch.Tensor
            Filter band tensor (0-5 for u,g,r,i,z,y filters).

        Returns
        -------
        torch.Tensor
            Aligned and cropped donut image tensor containing only high-quality
            donuts that pass the SNR threshold criteria.

        Notes
        -----
        This method performs the alignment portion of the full pipeline:
        1. Generates initial crop centers using a grid pattern
        2. Iteratively refines donut positions using AlignNet predictions
        3. Updates field coordinates based on pixel shifts
        4. Filters results based on signal-to-noise ratio (SNR > alpha)
        5. Returns only the aligned cropped images for further processing

        Unlike the full forward() method, this returns aligned images rather
        than Zernike coefficient predictions, making it suitable for detection
        and quality assessment workflows.
        """
        centers = get_centers(image[0, 0, :, :], self.CROP_SIZE).to(self.device_val)
        cropped_image = batched_crop(image[:, 0, :, :], centers, crop_size=self.CROP_SIZE).float()
        fx, fy, intra, band = fx.clone()[0, :, :].float(), fy.clone()[0, :, :].float(), intra.clone()[0, :, :].float(), band.clone()[0, :, :].int()

        for n in range(self.refinements):
            pixel_shifts = self.alignnet_model(cropped_image[:, 0, :, :],
                                               fx.clone(),
                                               fy.clone(), intra.clone(),
                                               band.clone()) * self.SCALE
            centers += pixel_shifts.int()

            cropped_image = batched_crop(image[0, :, :, :].float(),
                                         centers, crop_size=self.CROP_SIZE)

            fx += pixel_shifts[:, 0][..., None].int().float() * self.deg_per_pix
            fy += pixel_shifts[:, 1][..., None].int().float() * self.deg_per_pix

        SNR = self.single_conv_batched(cropped_image[:, 0, :, :])[..., None]

        keep_ind = SNR[:, 0] > self.alpha
        cropped_image = cropped_image[keep_ind]
        return cropped_image

    def forward(
        self,
        image: torch.Tensor,
        fx: torch.Tensor,
        fy: torch.Tensor,
        intra: torch.Tensor,
        band: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the Neural Active Optics System.

        Parameters
        ----------
        image : torch.Tensor
            Input donut image tensor of shape (batch_size, channels, height, width).
        fx : torch.Tensor
            Field x-coordinates tensor.
        fy : torch.Tensor
            Field y-coordinates tensor.
        intra : torch.Tensor
            Intra/extra-focal indicator tensor (0 for intra, 1 for extra).
        band : torch.Tensor
            Filter band tensor (0-5 for u,g,r,i,z,y filters).

        Returns
        -------
        torch.Tensor
            Predicted Zernike coefficients tensor.

        Notes
        -----
        This method performs the complete pipeline:
        1. Crops donut images from full field
        2. Iteratively refines alignment using AlignNet
        3. Filters donuts based on SNR criteria
        4. Predicts Zernike coefficients using WaveNet
        5. Aggregates predictions using AggregatorNet (if enabled)
        """
        centers = get_centers(image[0, 0, :, :], self.CROP_SIZE).to(self.device_val)
        cropped_image = batched_crop(image[:, 0, :, :], centers, crop_size=self.CROP_SIZE).float()

        fx, fy, intra, band = fx.clone()[0, :, :].float(), fy.clone()[0, :, :].float(), intra.clone()[0, :, :].float(), band.clone()[0, :, :].int()

        pixel_shifts = self.alignnet_model(
            cropped_image[:, 0, :, :], fx.clone(), fy.clone(), intra.clone(), band.clone()
        ) * self.SCALE
        centers += pixel_shifts.int()

        cropped_image = batched_crop(
            image[0, :, :, :].float(), centers, crop_size=self.CROP_SIZE
        )

        fx += pixel_shifts[:, 0][..., None].int().float() * self.deg_per_pix
        fy += pixel_shifts[:, 1][..., None].int().float() * self.deg_per_pix

        SNR = self.single_conv_batched(cropped_image[:, 0, :, :])[..., None]

        keep_ind = SNR[:, 0] > self.alpha
        self.cropped_image = copy.deepcopy(cropped_image)
        cropped_image = cropped_image[keep_ind]

        fx = fx[keep_ind]
        fy = fy[keep_ind]
        intra = intra[keep_ind]
        band = band[keep_ind]
        SNR = SNR[keep_ind]
        total_zernikes = self.wavenet_model(cropped_image[:, 0, :, :], fx.clone(), fy.clone(),
                                            intra.clone(), band.clone())
        total_zernikes = total_zernikes/1000
        # Ensure all tensors are on the same device before concatenation
        device = total_zernikes.device

        # Move all tensors to the same device as total_zernikes
        fx = fx.to(device)
        fy = fy.to(device)
        SNR = SNR.to(device)
        self.total_zernikes = total_zernikes
        # Match training data normalization: use local max like in dataloader
        # Training data: snr = torch.tensor(loaded_data["snr"]).to(self.device)[..., None] / torch.tensor(loaded_data["snr"]).max()
        # Note: Training data uses global max across all data, but we only have local data
        # Using local max with epsilon for numerical stability
        SNR_normalized = SNR / (SNR.max() + 1e-8)  # Add small epsilon to avoid division by zero

        # Match training data position processing: concatenate field positions
        # Training data: position = torch.concatenate([field_x, field_y], axis=-1).to(self.device)
        position = torch.cat([fx, fy], dim=-1)

        # Concatenate features in the same order as training data: [zernikes, position, snr]
        embedded_features = torch.cat([
            total_zernikes,
            position,
            SNR_normalized
        ], dim=1)

        if embedded_features.shape[0] > self.max_seq_length:
            embedded_features = embedded_features[:self.max_seq_length, :]
        else:
            padding = torch.zeros((self.max_seq_length - embedded_features.shape[0],
                                   embedded_features.shape[1])).to(self.device_val)
            embedded_features = torch.cat([embedded_features,
                                           padding], axis=0).to(self.device_val).float()

        embedded_features = embedded_features[None, ...]

        # Match training data mean computation:
        # Training data: zk_mean1 = torch.mean(zk_pred1, dim=0) / 1000
        # Since total_zernikes is already divided by 1000, we don't divide again
        mean_zernike = torch.mean(total_zernikes, axis=0)

        # (OPTIONAL) Check the types
        if self.aggregator_on:
            # Match training data: mean_zernike should NOT have convert_zernikes_deploy applied
            # Training data stores mean directly without conversion
            final_zernike_prediction = self.aggregatornet_model((embedded_features, mean_zernike))
        else:
            final_zernike_prediction = mean_zernike
        final_zernike_prediction = self.final_layer(final_zernike_prediction)

        return final_zernike_prediction

    def training_step(self, batch, batch_idx):
        """Perform a single training step.

        Parameters
        ----------
        batch : tuple
            Training batch containing input tensors and target Zernike coefficients.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Computed training loss for the batch.
        """
        (image, fx, fy, intra, band), y = batch  # y is the target token
        logits = self.forward(image, fx, fy, intra, band)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Perform a single validation step.

        Parameters
        ----------
        batch : tuple
            Validation batch containing input tensors and target Zernike coefficients.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Computed validation loss for the batch.
        """
        (image, fx, fy, intra, band), y = batch  # y is the target token
        logits = self.forward(image, fx, fy, intra, band)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def loss_fn(self, x, y):
        """Compute the Root Sum of Squared Errors (mRSSE) loss.

        Parameters
        ----------
        x : torch.Tensor
            Predicted Zernike coefficients.
        y : torch.Tensor
            Target Zernike coefficients.

        Returns
        -------
        torch.Tensor
            Mean Root Sum of Squared Errors loss value.
        """
        sse = F_loss.mse_loss(x, y, reduction="none").sum(dim=-1)
        mRSSe = torch.sqrt(sse).mean()
        return mRSSe

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer),
                "monitor": "val_loss",
                "frequency": 1,
            },
        }

    def deploy_run(self, exposure, detectorName=None):
        """Run inference on a real LSST exposure for deployment.

        Parameters
        ----------
        exposure : lsst.afw.image.Exposure
            LSST exposure object containing the raw donut image data.
        detectorName : str, optional
            Name of the detector/chip. If None, uses the CHIPID from exposure metadata.

        Returns
        -------
        torch.Tensor
            Predicted Zernike coefficients for the exposure.

        Notes
        -----
        This method performs the complete inference pipeline:
        1. Assembles CCD from raw exposure
        2. Subtracts background
        3. Extracts metadata (filter, focal plane position)
        4. Computes field coordinates for all detected donuts
        5. Runs forward pass to predict Zernike coefficients
        """
        camera = LsstCam().getCamera()
        assembleCcdTask = AssembleCcdTask()
        new = assembleCcdTask.assembleCcd(exposure)
        SubtractBackground = subtractBackground.SubtractBackgroundTask()
        SubtractBackground.run(new)
        image = new.getImage().array
        header = exposure.metadata
        filter_name = header['FILTER']
        if detectorName is None:
            detectorName = header['CHIPID']
        #  U G R I Z Y
        if 'u' in filter_name:
            filter_name = torch.tensor([0])
        elif 'g' in filter_name:
            filter_name = torch.tensor([1])
        elif 'r' in filter_name:
            filter_name = torch.tensor([2])
        elif 'i' in filter_name:
            filter_name = torch.tensor([3])
        elif 'z' in filter_name:
            filter_name = torch.tensor([4])
        elif 'y' in filter_name:
            filter_name = torch.tensor([5])

        # check if intra or extra
        if header['CCDSLOT'][2:] == '1':
            focal = torch.tensor([0]).float()
        elif header['CCDSLOT'][2:] == '0':
            focal = torch.tensor([1]).float()

        centers = get_centers(image, self.CROP_SIZE)

        # Cache camera transforms to avoid repeated instantiation
        camera_transforms = LsstCameraTransforms(camera)

        # Vectorized field coordinate computation
        field_coords = []
        for x, y in centers:
            coord = camera_transforms.ccdPixelToFocalMm(x, y, detectorName=detectorName)
            field_coords.append([coord[0] / self.mm_pix * self.deg_per_pix, coord[1] / self.mm_pix * self.deg_per_pix])

        field_coords = torch.tensor(field_coords, dtype=torch.float32)
        field_x = field_coords[:, 0].unsqueeze(1).to(self.device_val)[None, ...]
        field_y = field_coords[:, 1].unsqueeze(1).to(self.device_val)[None, ...]

        # Vectorized focal and band values - ensure correct shape for forward method
        focal_val = focal.expand(centers.shape[0], 1).to(self.device_val)[None, ...]
        band_val = filter_name.expand(centers.shape[0], 1).to(self.device_val)[None, ...]

        image_tensor = F.to_tensor(image)[None, ...]
        with torch.no_grad():
            pred = self.forward(image_tensor, field_x, field_y, focal_val, band_val)
        return pred

    def deploy_detect(self, exposure, detectorName=None):
        """Run inference on a real LSST exposure for deployment.

        Parameters
        ----------
        exposure : lsst.afw.image.Exposure
            LSST exposure object containing the raw donut image data.
        detectorName : str, optional
            Name of the detector/chip. If None, uses the CHIPID from exposure metadata.

        Returns
        -------
        torch.Tensor
            Predicted Zernike coefficients for the exposure.

        Notes
        -----
        This method performs the complete inference pipeline:
        1. Assembles CCD from raw exposure
        2. Subtracts background
        3. Extracts metadata (filter, focal plane position)
        4. Computes field coordinates for all detected donuts
        5. Runs forward pass to predict Zernike coefficients
        """
        camera = LsstCam().getCamera()
        assembleCcdTask = AssembleCcdTask()
        new = assembleCcdTask.assembleCcd(exposure)
        SubtractBackground = subtractBackground.SubtractBackgroundTask()
        SubtractBackground.run(new)
        image = new.getImage().array
        header = exposure.metadata
        filter_name = header['FILTER']
        if detectorName is None:
            detectorName = header['CHIPID']
        #  U G R I Z Y
        if 'u' in filter_name:
            filter_name = torch.tensor([0])
        elif 'g' in filter_name:
            filter_name = torch.tensor([1])
        elif 'r' in filter_name:
            filter_name = torch.tensor([2])
        elif 'i' in filter_name:
            filter_name = torch.tensor([3])
        elif 'z' in filter_name:
            filter_name = torch.tensor([4])
        elif 'y' in filter_name:
            filter_name = torch.tensor([5])

        # check if intra or extra
        if header['CCDSLOT'][2:] == '1':
            focal = torch.tensor([0]).float()
        elif header['CCDSLOT'][2:] == '0':
            focal = torch.tensor([1]).float()

        centers = get_centers(image, self.CROP_SIZE)

        # Cache camera transforms to avoid repeated instantiation
        camera_transforms = LsstCameraTransforms(camera)

        # Vectorized field coordinate computation
        field_coords = []
        for x, y in centers:
            coord = camera_transforms.ccdPixelToFocalMm(x, y, detectorName=detectorName)
            field_coords.append([coord[0] / self.mm_pix * self.deg_per_pix, coord[1] / self.mm_pix * self.deg_per_pix])

        field_coords = torch.tensor(field_coords, dtype=torch.float32)
        field_x = field_coords[:, 0].unsqueeze(1).to(self.device_val)[None, ...]
        field_y = field_coords[:, 1].unsqueeze(1).to(self.device_val)[None, ...]

        # Vectorized focal and band values - ensure correct shape for forward_align method
        focal_val = focal.expand(centers.shape[0], 1).to(self.device_val)[None, ...]
        band_val = filter_name.expand(centers.shape[0], 1).to(self.device_val)[None, ...]

        image_tensor = F.to_tensor(image)[None, ...]
        with torch.no_grad():
            aligned_images = self.forward_align(image_tensor.to(self.device_val), field_x, field_y, focal_val, band_val)
        return aligned_images
