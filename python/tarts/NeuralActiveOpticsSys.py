"""Neural network to predict zernike coefficients from donut images and positions."""

# Standard library imports
import copy
import logging
import os
from typing import Any, Dict, List

# Third-party imports
import joblib
import pytorch_lightning as pl
import torch
import torch.nn.functional as F_loss
import torchvision.transforms.functional as F
import yaml
from lsst.ip.isr import AssembleCcdTask
from lsst.meas.algorithms import subtractBackground
from lsst.obs.lsst import LsstCam
from lsst.obs.lsst.cameraTransforms import LsstCameraTransforms
from torch import nn, vmap
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Local/application imports
from .aggregatornet import AggregatorNet
from .constants import MAP_DETECTOR_TO_NUMBER
from .lightning_alignnet import AlignNetSystem
from .lightning_wavenet import WaveNetSystem
from .utils import (
    batched_crop,
    convert_zernikes_deploy,
    get_centers,
    single_conv,
    shift_offcenter,
)

logger = logging.getLogger(__name__)


class NeuralActiveOpticsSys(pl.LightningModule):
    """Transfer learning driven WaveNet."""

    def __init__(
        self,
        dataset_params,
        wavenet_path=None,
        alignet_path=None,
        aggregatornet_path=None,
        lr=1e-3,
        final_layer=None,
        aggregator_on=True,
        pretrained=True,
        compile_models=False,
        ood_model_path=None,
        kmin=None,
        kmax=float("inf"),
        sigma_frac=0.1,
    ) -> None:
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
        pretrained : bool, optional
            Whether to use pre-trained CNN weights when creating new models (when checkpoint paths are None).
            Set to False for deployment mode to avoid downloading weights. Defaults to True.
            Note: This parameter is ignored when loading from checkpoint files.
        compile_models : bool, optional
            Whether to apply torch.compile to the submodels (WaveNet, AlignNet, AggregatorNet).
            This can significantly speed up inference but may increase compilation time on first run.
            Automatically selects backend: "inductor" for CPU, default for GPU.
            Defaults to False.
        ood_model_path : str, optional
            Path to OOD detection model (joblib file). If provided, OOD detection will be performed
            during inference and scores will be stored in internal metadata. Defaults to None.
        kmin : float, optional
            Minimum frequency magnitude for frequency filtering. If None, frequency filtering is disabled.
            If set, applies frequency filter to each cropped image before passing to WaveNet.
            Defaults to None.
        kmax : float, optional
            Maximum frequency magnitude for frequency filtering. Only used if kmin is not None.
            Defaults to float('inf').
        sigma_frac : float, optional
            Fractional Gaussian width for smooth filter edges in frequency filtering.
            Only used if kmin is not None. Defaults to 0.1.
        """
        super().__init__()
        self.save_hyperparameters()
        self.device_val = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize frequency filtering parameters
        self.kmin = kmin
        self.kmax = kmax
        self.sigma_frac = sigma_frac

        # Initialize attributes that may be set during forward pass
        # These can be lists or tensors depending on context
        self.fx: List[torch.Tensor] | torch.Tensor = []
        self.fy: List[torch.Tensor] | torch.Tensor = []
        self.intra: List[torch.Tensor] | torch.Tensor = []
        self.band: List[torch.Tensor] | torch.Tensor = []
        self.SNR: List[torch.Tensor] | torch.Tensor = []
        self.centers: List[torch.Tensor] | torch.Tensor = []
        self.total_zernikes: torch.Tensor | None = None
        self.cropped_image: torch.Tensor | None = None
        self.ood_scores: torch.Tensor | None = None

        # Load OOD detection model if path is provided
        self.ood_model = None
        self.ood_mean = None
        if ood_model_path is not None and os.path.exists(ood_model_path):
            logger.info(f"Loading OOD detection model from {ood_model_path}...")
            ood_data = joblib.load(ood_model_path)
            self.ood_model = ood_data["cov_model"]
            self.ood_mean = torch.tensor(ood_data["mean"], device=self.device_val, dtype=torch.float32)
            logger.info("OOD detection model loaded successfully")
        elif ood_model_path is not None:
            logger.warning(f"OOD model path provided but file not found: {ood_model_path}")
            logger.info("Continuing without OOD detection...")

        # Load parameters from YAML file once
        with open(dataset_params, "r") as yaml_file:
            params = yaml.safe_load(yaml_file)

        if wavenet_path is None:
            self.wavenet_model = WaveNetSystem(pretrained=pretrained).to(self.device_val)
        else:
            # Always use checkpoint loading - the pretrained parameter doesn't matter
            # when loading from checkpoint
            self.wavenet_model = WaveNetSystem.load_from_checkpoint(
                wavenet_path, map_location=str(self.device_val)
            ).to(self.device_val)

        if alignet_path is None:
            self.alignnet_model = AlignNetSystem(pretrained=pretrained).to(self.device_val)
        else:
            # Always use checkpoint loading - the pretrained parameter doesn't matter
            # when loading from checkpoint
            self.alignnet_model = AlignNetSystem.load_from_checkpoint(
                alignet_path, map_location=str(self.device_val)
            ).to(self.device_val)

        self.max_seq_length = params["max_seq_len"]
        self.aggregator_on = aggregator_on
        logger.debug(f"Using device: {self.device_val}")
        if aggregatornet_path is None:
            d_model = params["aggregator_model"]["d_model"]
            nhead = params["aggregator_model"]["nhead"]
            num_layers = params["aggregator_model"]["num_layers"]
            dim_feedforward = params["aggregator_model"]["dim_feedforward"]
            self.aggregatornet_model = AggregatorNet(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                max_seq_length=self.max_seq_length,
            ).to(self.device_val)
        else:
            self.aggregatornet_model = AggregatorNet.load_from_checkpoint(aggregatornet_path).to(
                self.device_val
            )

        if final_layer is not None:
            layers = [
                nn.ReLU(),
                nn.Linear(17, final_layer),  # WaveNet outputs 17 Zernike coefficients
            ]
            self.final_layer = nn.Sequential(*layers)
        else:
            self.final_layer = self.identity

        # Use already loaded params
        self.refinements = params["refinements"]
        self.CROP_SIZE = params["CROP_SIZE"]
        self.mm_pix = params["mm_pix"]
        self.deg_per_pix = params["deg_per_pix"]
        self.alpha = params["alpha"]
        self.SCALE = params["adjustment_AlignNet"]
        self.num_zernikes = len(params["noll_zk"])

        # Apply torch.compile to submodels if requested
        if compile_models:
            # Determine compilation backend based on device
            if self.device_val.type == "cpu":
                compile_backend = "inductor"
                logger.info("Compiling submodels with torch.compile (CPU backend: inductor)...")
            else:
                compile_backend = None  # Use default backend for GPU
                logger.info("Compiling submodels with torch.compile (GPU backend: default)...")

            try:
                if compile_backend is not None:
                    self.wavenet_model = torch.compile(self.wavenet_model, backend=compile_backend)
                else:
                    self.wavenet_model = torch.compile(self.wavenet_model)
                logger.info(f"WaveNet compiled with backend: {compile_backend or 'default'}")
            except (RuntimeError, TypeError, AttributeError) as e:
                logger.warning(f"WaveNet compilation failed: {e}")

            try:
                if compile_backend is not None:
                    self.alignnet_model = torch.compile(self.alignnet_model, backend=compile_backend)
                else:
                    self.alignnet_model = torch.compile(self.alignnet_model)
                logger.info(f"AlignNet compiled with backend: {compile_backend or 'default'}")
            except (RuntimeError, TypeError, AttributeError) as e:
                logger.warning(f"AlignNet compilation failed: {e}")

            try:
                if compile_backend is not None:
                    self.aggregatornet_model = torch.compile(
                        self.aggregatornet_model, backend=compile_backend
                    )
                else:
                    self.aggregatornet_model = torch.compile(self.aggregatornet_model)
                logger.info(f"AggregatorNet compiled with backend: {compile_backend or 'default'}")
            except (RuntimeError, TypeError, AttributeError) as e:
                logger.warning(f"AggregatorNet compilation failed: {e}")

            logger.info("Model compilation completed!")
        else:
            logger.info("Model compilation disabled (compile_models=False)")

    def compile_submodels(self):
        """Apply torch.compile to all submodels for faster inference.

        This method can be called after model initialization to enable compilation.
        Useful for deployment scenarios where you want to optimize inference speed.
        Automatically selects the appropriate backend based on the device (CPU: inductor, GPU: default).
        """
        # Determine compilation backend based on device
        if self.device_val.type == "cpu":
            compile_backend = "inductor"
            logger.info("Compiling submodels with torch.compile (CPU backend: inductor)...")
        else:
            compile_backend = None  # Use default backend for GPU
            logger.info("Compiling submodels with torch.compile (GPU backend: default)...")

        try:
            if compile_backend is not None:
                self.wavenet_model = torch.compile(self.wavenet_model, backend=compile_backend)
            else:
                self.wavenet_model = torch.compile(self.wavenet_model)
            logger.info(f"WaveNet compiled with backend: {compile_backend or 'default'}")
        except (RuntimeError, TypeError, AttributeError) as e:
            logger.warning(f"WaveNet compilation failed: {e}")

        try:
            if compile_backend is not None:
                self.alignnet_model = torch.compile(self.alignnet_model, backend=compile_backend)
            else:
                self.alignnet_model = torch.compile(self.alignnet_model)
            logger.info(f"AlignNet compiled with backend: {compile_backend or 'default'}")
        except (RuntimeError, TypeError, AttributeError) as e:
            logger.warning(f"AlignNet compilation failed: {e}")

        try:
            if compile_backend is not None:
                self.aggregatornet_model = torch.compile(self.aggregatornet_model, backend=compile_backend)
            else:
                self.aggregatornet_model = torch.compile(self.aggregatornet_model)
            logger.info(f"AggregatorNet compiled with backend: {compile_backend or 'default'}")
        except (RuntimeError, TypeError, AttributeError) as e:
            logger.warning(f"AggregatorNet compilation failed: {e}")

        logger.info("Model compilation completed!")

    def get_internal_data(self):
        """Retrieve all internal data as a list of dictionaries.

        Each dictionary contains the data for one donut/crop:
        - cropped_image: The cropped image tensor
        - fx: Field x position
        - fy: Field y position
        - intra: Intra-focal measurement
        - band: Band information
        - SNR: Signal-to-noise ratio
        - centers: Center coordinates
        - zernikes: Estimated Zernike coefficients
        - ood_score: Out-of-distribution score (if OOD detection is enabled)

        Returns:
        --------
        list of dict
            List of dictionaries, each containing the data for one donut.
            Returns empty list if no data is available.
        """
        internal_data: List[Dict[str, Any]] = []

        # Check if we have valid data (not NaN-filled)
        if len(self.fx) > 0 and isinstance(self.fx, list) and not torch.isnan(self.fx[0]):
            num_donuts = len(self.fx)

            # Ensure these are not None before indexing
            if self.cropped_image is None or self.total_zernikes is None:
                return internal_data

            for i in range(num_donuts):
                data_dict = {
                    "cropped_image": self.cropped_image[i].clone().detach(),
                    "fx": self.fx[i].clone().detach(),
                    "fy": self.fy[i].clone().detach(),
                    "intra": self.intra[i].clone().detach(),
                    "band": self.band[i].clone().detach(),
                    "SNR": self.SNR[i].clone().detach(),
                    "centers": self.centers[i].clone().detach(),
                    "zernikes": self.total_zernikes[i].clone().detach(),
                }
                # Add OOD score if available
                if self.ood_scores is not None and i < len(self.ood_scores):
                    data_dict["ood_score"] = self.ood_scores[i].clone().detach()
                else:
                    data_dict["ood_score"] = None

                internal_data.append(data_dict)

        return internal_data

    def filter_image_by_frequency_torch(self, image, kmin=0, kmax=float("inf"), sigma_frac=0.1, device=None):
        """Apply a smooth Gaussian-weighted frequency filter in Fourier space to a 2D image.

        Uses PyTorch for GPU acceleration.

        Parameters
        ----------
        image : torch.Tensor or np.ndarray
            2D input image (real-valued). Shape: (H, W)
        kmin, kmax : float
            Minimum and maximum |k| (frequency magnitude) to keep.
        sigma_frac : float, optional
            Fractional Gaussian width for smooth filter edges.
            Higher = softer transitions (default 0.1).
        show : bool, optional
            If True, visualize FFT magnitude, filter mask, and filtered image.
        device : str or torch.device, optional
            Device to run on ('cpu' or 'cuda'). Defaults to CUDA if available.

        Returns
        -------
        filtered_image : torch.Tensor
            The reconstructed real-space image after filtering.
        weight : torch.Tensor
            The Fourier-space weighting mask.
        """
        # --- Setup device ---
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Convert image to float tensor ---
        if not torch.is_tensor(image):
            image = torch.tensor(image, dtype=torch.float32)
        image = image.to(device)

        # --- Compute 2D FFT ---
        f = torch.fft.fft2(image)
        fshift = torch.fft.fftshift(f)

        # --- Build frequency grids ---
        ny, nx = image.shape
        kx = torch.fft.fftshift(torch.fft.fftfreq(nx, device=device))
        ky = torch.fft.fftshift(torch.fft.fftfreq(ny, device=device))
        KX, KY = torch.meshgrid(kx, ky, indexing="xy")
        K_mag = torch.sqrt(KX**2 + KY**2)

        # --- Smooth Gaussian-weighted filter ---
        kmax_valid = K_mag.max()
        kmax = min(kmax, kmax_valid.item())
        sigma = sigma_frac * (kmax - kmin)
        sigma = max(sigma, 1e-6)  # avoid division by zero

        low_edge = 1 / (1 + torch.exp(-(K_mag - kmin) / sigma))
        high_edge = 1 / (1 + torch.exp((K_mag - kmax) / sigma))
        weight = low_edge * high_edge

        # --- Apply filter ---
        f_filtered = fshift * weight

        # --- Inverse FFT ---
        reconstructed = torch.fft.ifft2(torch.fft.ifftshift(f_filtered)).real
        return reconstructed, weight

    def filter_batch_images(self, images):
        """Apply frequency filtering to a batch of 2D images.

        Parameters
        ----------
        images : torch.Tensor
            Batch of 2D images. Shape: (batch_size, height, width)

        Returns
        -------
        filtered_images : torch.Tensor
            Batch of filtered 2D images. Shape: (batch_size, height, width)
        """
        if self.kmin is None:
            # Frequency filtering is disabled
            return images

        filtered_images = []
        for i in range(images.shape[0]):
            filtered_img, _ = self.filter_image_by_frequency_torch(
                images[i], kmin=self.kmin, kmax=self.kmax, sigma_frac=self.sigma_frac, device=self.device_val
            )
            filtered_images.append(filtered_img)

        return torch.stack(filtered_images, dim=0)

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

        def _single_conv_wrapper(x):
            """Wrapper function for single_conv to use with vmap."""
            return single_conv(x, device=str(device))

        return vmap(_single_conv_wrapper)(data)

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
        device_str = "cpu" if data.device.type == "cpu" else "cuda"
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
        fx, fy, intra, band = (
            fx.clone()[0, :, :].float(),
            fy.clone()[0, :, :].float(),
            intra.clone()[0, :, :].float(),
            band.clone()[0, :, :].int(),
        )

        for n in range(self.refinements):
            pixel_shifts = (
                self.alignnet_model(
                    cropped_image[:, 0, :, :], fx.clone(), fy.clone(), intra.clone(), band.clone()
                )
                * self.SCALE
            )
            centers += pixel_shifts.int()

            cropped_image = batched_crop(image[0, :, :, :].float(), centers, crop_size=self.CROP_SIZE)

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

        fx, fy, intra, band = (
            fx.clone()[0, :, :].float(),
            fy.clone()[0, :, :].float(),
            intra.clone()[0, :, :].float(),
            band.clone()[0, :, :].int(),
        )

        pixel_shifts = (
            self.alignnet_model(
                cropped_image[:, 0, :, :], fx.clone(), fy.clone(), intra.clone(), band.clone()
            )
            * self.SCALE
        )
        centers += pixel_shifts.int()

        cropped_image = batched_crop(image[0, :, :, :].float(), centers, crop_size=self.CROP_SIZE)

        fx += pixel_shifts[:, 0][..., None].int().float() * self.deg_per_pix
        fy += pixel_shifts[:, 1][..., None].int().float() * self.deg_per_pix

        SNR = self.single_conv_batched(cropped_image[:, 0, :, :])[..., None]

        keep_ind = SNR[:, 0] > self.alpha

        # Check if any donuts remain after SNR filtering
        if keep_ind.sum() == 0:
            # No donuts detected, return zeros for Zernike coefficients
            # The final_layer will process the output, so we need to return the expected size
            # If final_layer is present, return the size it outputs
            self.fx = [torch.tensor(0)]
            self.fy = [torch.tensor(0)]
            self.intra = [torch.tensor(0)]
            self.band = [torch.tensor(0)]
            self.SNR = [torch.tensor(0)]
            self.centers = [torch.tensor([0, 0])]
            self.total_zernikes = torch.zeros((1, self.num_zernikes), device=self.device_val)
            self.cropped_image = torch.zeros((1, self.CROP_SIZE, self.CROP_SIZE), device=self.device_val)
            self.total_zernikes = torch.zeros((1, self.num_zernikes), device=self.device_val)
            self.ood_scores = torch.tensor([float("nan")], device=self.device_val)
            return torch.zeros((1, self.num_zernikes), device=self.device_val)

        cropped_image = cropped_image[keep_ind]

        fx = fx[keep_ind]
        fy = fy[keep_ind]
        intra = intra[keep_ind]
        band = band[keep_ind]
        SNR = SNR[keep_ind]

        # Apply frequency filtering if kmin is set
        filtered_cropped_image = self.filter_batch_images(cropped_image[:, 0, :, :])

        total_zernikes = self.wavenet_model(
            filtered_cropped_image, fx.clone(), fy.clone(), intra.clone(), band.clone()
        )
        total_zernikes = total_zernikes / 1000

        # Compute OOD scores if OOD model is available
        ood_scores = None
        if self.ood_model is not None and self.wavenet_model.wavenet.predictor_features is not None:
            # Get predictor penultimate features and detach/convert to CPU for numpy
            penultimate = self.wavenet_model.wavenet.predictor_features  # Shape: (batch_size, n_features)

            # Detach from computation graph and move to CPU for numpy conversion
            features_np = penultimate.detach().cpu().numpy()
            if self.ood_mean is not None:
                features_centered = features_np - self.ood_mean.cpu().numpy()
                mahalanobis_dist = self.ood_model.mahalanobis(features_centered)
                # Store as tensor
                ood_scores = torch.tensor(mahalanobis_dist, device=self.device_val, dtype=torch.float32)

        # Ensure all tensors are on the same device before concatenation
        device = total_zernikes.device

        # Move all tensors to the same device as total_zernikes
        fx = fx.to(device)
        fy = fy.to(device)
        SNR = SNR.to(device)
        self.centers = centers[keep_ind]
        # These attributes can be either lists or tensors depending on context
        # Suppress type checking for these dynamic attribute assignments
        self.fx = fx
        self.fy = fy
        self.intra = intra
        self.band = band
        self.SNR = SNR
        self.total_zernikes = total_zernikes
        self.cropped_image = cropped_image[:, 0, :, :]
        self.ood_scores = ood_scores
        # Match training data normalization: use local max like in dataloader
        # Training data: snr = torch.tensor(loaded_data["snr"]).to(self.device)[..., None]
        # / torch.tensor(loaded_data["snr"]).max()
        # Note: Training data uses global max across all data, but we only have local data
        # Using local max with epsilon for numerical stability
        SNR_normalized = SNR / (SNR.max() + 1e-8)  # Add small epsilon to avoid division by zero

        # Match training data position processing: concatenate field positions
        # Training data: position = torch.concatenate([field_x, field_y], axis=-1).to(self.device)
        position = torch.cat([fx, fy], dim=-1)

        # Concatenate features in the same order as training data: [zernikes, position, snr]
        embedded_features = torch.cat([total_zernikes, position, SNR_normalized], dim=1)

        if embedded_features.shape[0] > self.max_seq_length:
            embedded_features = embedded_features[: self.max_seq_length, :]
        else:
            padding = torch.zeros(
                (self.max_seq_length - embedded_features.shape[0], embedded_features.shape[1])
            ).to(self.device_val)
            embedded_features = torch.cat([embedded_features, padding], dim=0).to(self.device_val).float()

        embedded_features = embedded_features[None, ...]

        # Match training data mean computation:
        # Training data: zk_mean1 = torch.mean(zk_pred1, dim=0) / 1000
        # Since total_zernikes is already divided by 1000, we don't divide again
        mean_zernike = torch.mean(total_zernikes, dim=0)

        # (OPTIONAL) Check the types
        if self.aggregator_on:
            # Match training data: mean_zernike should NOT have convert_zernikes_deploy applied
            # Training data stores mean directly without conversion
            final_zernike_prediction = self.aggregatornet_model((embedded_features, mean_zernike))
        else:
            final_zernike_prediction = mean_zernike
        final_zernike_prediction = self.final_layer(final_zernike_prediction)

        return final_zernike_prediction

    def forward_shifts(
        self,
        image: torch.Tensor,
        fx: torch.Tensor,
        fy: torch.Tensor,
        intra: torch.Tensor,
        band: torch.Tensor,
        shift_amount: int = 5,
    ) -> torch.Tensor:
        """Forward pass with random image shifts applied before WaveNet.

        This method is similar to forward() but applies random shifts to cropped images
        before passing them to WaveNet, which can help with robustness and data augmentation.

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
        shift_amount : int, default=5
            Maximum amount of random shift to apply to cropped images.

        Returns
        -------
        torch.Tensor
            Predicted Zernike coefficients tensor.

        Notes
        -----
        This method performs the same pipeline as forward() but with random shifts:
        1. Crops donut images from full field
        2. Iteratively refines alignment using AlignNet
        3. Filters donuts based on SNR criteria
        4. Applies random shifts to cropped images using shift_offcenter
        5. Predicts Zernike coefficients using WaveNet
        6. Aggregates predictions using AggregatorNet (if enabled)
        """
        centers = get_centers(image[0, 0, :, :], self.CROP_SIZE).to(self.device_val)
        cropped_image = batched_crop(image[:, 0, :, :], centers, crop_size=self.CROP_SIZE).float()

        fx, fy, intra, band = (
            fx.clone()[0, :, :].float(),
            fy.clone()[0, :, :].float(),
            intra.clone()[0, :, :].float(),
            band.clone()[0, :, :].int(),
        )

        pixel_shifts = (
            self.alignnet_model(
                cropped_image[:, 0, :, :], fx.clone(), fy.clone(), intra.clone(), band.clone()
            )
            * self.SCALE
        )
        centers += pixel_shifts.int()

        cropped_image = batched_crop(image[0, :, :, :].float(), centers, crop_size=self.CROP_SIZE)

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

        # Apply random shifts to cropped images before WaveNet
        shifted_images = []
        for i in range(cropped_image.shape[0]):
            shifted_img = shift_offcenter(cropped_image[i, 0, :, :], adjust=shift_amount, return_offset=False)
            shifted_images.append(shifted_img)

        # Stack shifted images back into a batch
        shifted_cropped_image = torch.stack(shifted_images, dim=0)

        # Apply frequency filtering if kmin is set
        filtered_shifted_cropped_image = self.filter_batch_images(shifted_cropped_image)

        total_zernikes = self.wavenet_model(
            filtered_shifted_cropped_image, fx.clone(), fy.clone(), intra.clone(), band.clone()
        )
        total_zernikes = total_zernikes / 1000

        # Compute OOD scores if OOD model is available
        ood_scores = None
        if self.ood_model is not None and self.wavenet_model.wavenet.predictor_features is not None:
            # Get predictor penultimate features and detach/convert to CPU for numpy
            penultimate = self.wavenet_model.wavenet.predictor_features  # Shape: (batch_size, n_features)

            # Detach from computation graph and move to CPU for numpy conversion
            features_np = penultimate.detach().cpu().numpy()
            if self.ood_mean is not None:
                features_centered = features_np - self.ood_mean.cpu().numpy()
                mahalanobis_dist = self.ood_model.mahalanobis(features_centered)

                # Store as tensor
                ood_scores = torch.tensor(mahalanobis_dist, device=self.device_val, dtype=torch.float32)

        # Ensure all tensors are on the same device before concatenation
        device = total_zernikes.device

        # Move all tensors to the same device as total_zernikes
        fx = fx.to(device)
        fy = fy.to(device)
        SNR = SNR.to(device)
        self.total_zernikes = total_zernikes
        self.ood_scores = ood_scores
        # Match training data normalization: use local max like in dataloader
        # Training data: snr = torch.tensor(loaded_data["snr"]).to(self.device)[..., None]
        # / torch.tensor(loaded_data["snr"]).max()
        # Note: Training data uses global max across all data, but we only have local data
        # Using local max with epsilon for numerical stability
        SNR_normalized = SNR / (SNR.max() + 1e-8)  # Add small epsilon to avoid division by zero

        # Match training data position processing: concatenate field positions
        # Training data: position = torch.concatenate([field_x, field_y], axis=-1).to(self.device)
        position = torch.cat([fx, fy], dim=-1)

        # Concatenate features in the same order as training data: [zernikes, position, snr]
        embedded_features = torch.cat([total_zernikes, position, SNR_normalized], dim=1)

        if embedded_features.shape[0] > self.max_seq_length:
            embedded_features = embedded_features[: self.max_seq_length, :]
        else:
            padding = torch.zeros(
                (self.max_seq_length - embedded_features.shape[0], embedded_features.shape[1])
            ).to(self.device_val)
            embedded_features = torch.cat([embedded_features, padding], dim=0).to(self.device_val).float()

        embedded_features = embedded_features[None, ...]

        # Match training data mean computation:
        # Training data: zk_mean1 = torch.mean(zk_pred1, dim=0) / 1000
        # Since total_zernikes is already divided by 1000, we don't divide again
        mean_zernike = torch.mean(total_zernikes, dim=0)

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

    def configure_optimizers(self) -> Any:
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
        try:
            assembleCcdTask = AssembleCcdTask()
            new = assembleCcdTask.assembleCcd(exposure)
            SubtractBackground = subtractBackground.SubtractBackgroundTask()
            SubtractBackground.run(new)
        except (AttributeError, RuntimeError, ValueError) as e:
            logger.warning(f"Switching to no CCD assembly: {e}")
            new = exposure
            SubtractBackground = subtractBackground.SubtractBackgroundTask()
            SubtractBackground.run(new)
        image = new.getImage().array
        header = exposure.metadata
        filter_name = header["FILTER"]
        if detectorName is None:
            full_detectorName = header["RAFTBAY"] + "_" + header["CCDSLOT"]
            detectorName = MAP_DETECTOR_TO_NUMBER[full_detectorName]
        #  U G R I Z Y
        if "u" in filter_name:
            filter_name = torch.tensor([0])
        elif "g" in filter_name:
            filter_name = torch.tensor([1])
        elif "r" in filter_name:
            filter_name = torch.tensor([2])
        elif "i" in filter_name:
            filter_name = torch.tensor([3])
        elif "z" in filter_name:
            filter_name = torch.tensor([4])
        elif "y" in filter_name:
            filter_name = torch.tensor([5])

        # check if intra or extra
        if header["CCDSLOT"][2:] == "1":
            focal = torch.tensor([0]).float()
        elif header["CCDSLOT"][2:] == "0":
            focal = torch.tensor([1]).float()

        centers = get_centers(image, self.CROP_SIZE)

        # Cache camera transforms to avoid repeated instantiation
        camera_transforms = LsstCameraTransforms(camera)

        # Vectorized field coordinate computation
        field_coords = []
        for x, y in centers:
            coord = camera_transforms.ccdPixelToFocalMm(x, y, detectorName=detectorName)
            field_coords.append(
                [
                    coord[0] / self.mm_pix * self.deg_per_pix,
                    coord[1] / self.mm_pix * self.deg_per_pix,
                ]
            )

        field_coords_tensor: torch.Tensor = torch.tensor(field_coords, dtype=torch.float32)
        # Extract field coordinates and add batch dimension
        field_x = field_coords_tensor[:, 0].unsqueeze(1).to(self.device_val)[None, ...]
        field_y = field_coords_tensor[:, 1].unsqueeze(1).to(self.device_val)[None, ...]

        # Vectorized focal and band values - ensure correct shape for forward method
        focal_val = focal.expand(centers.shape[0], 1).to(self.device_val)[None, ...]
        band_val = filter_name.expand(centers.shape[0], 1).to(self.device_val)[None, ...]

        image_tensor = F.to_tensor(image)[None, ...]
        with torch.no_grad():
            pred = self.forward(image_tensor, field_x, field_y, focal_val, band_val)
        return pred

    def deploy_run_shifts(self, exposure, detectorName=None, shift_amount=5):
        """Run inference on a real LSST exposure for deployment with random image shifts.

        This method is identical to deploy_run() except it uses forward_shifts() to apply
        random shifts to the cropped images before WaveNet processing.

        Parameters
        ----------
        exposure : lsst.afw.image.Exposure
            LSST exposure object containing the raw donut image data.
        detectorName : str, optional
            Name of the detector/chip. If None, uses the CHIPID from exposure metadata.
        shift_amount : int, default=5
            Maximum amount of random shift to apply to cropped images.

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
        5. Runs forward_shifts pass to predict Zernike coefficients with random shifts
        """
        camera = LsstCam().getCamera()
        assembleCcdTask = AssembleCcdTask()
        new = assembleCcdTask.assembleCcd(exposure)
        SubtractBackground = subtractBackground.SubtractBackgroundTask()
        SubtractBackground.run(new)
        image = new.getImage().array
        header = exposure.metadata
        filter_name = header["FILTER"]
        if detectorName is None:
            detectorName = header["CHIPID"]
        #  U G R I Z Y
        if "u" in filter_name:
            filter_name = torch.tensor([0])
        elif "g" in filter_name:
            filter_name = torch.tensor([1])
        elif "r" in filter_name:
            filter_name = torch.tensor([2])
        elif "i" in filter_name:
            filter_name = torch.tensor([3])
        elif "z" in filter_name:
            filter_name = torch.tensor([4])
        elif "y" in filter_name:
            filter_name = torch.tensor([5])

        # check if intra or extra
        if header["CCDSLOT"][2:] == "1":
            focal = torch.tensor([0]).float()
        elif header["CCDSLOT"][2:] == "0":
            focal = torch.tensor([1]).float()

        centers = get_centers(image, self.CROP_SIZE)

        # Cache camera transforms to avoid repeated instantiation
        camera_transforms = LsstCameraTransforms(camera)

        # Vectorized field coordinate computation
        field_coords = []
        for x, y in centers:
            coord = camera_transforms.ccdPixelToFocalMm(x, y, detectorName=detectorName)
            field_coords.append(
                [
                    coord[0] / self.mm_pix * self.deg_per_pix,
                    coord[1] / self.mm_pix * self.deg_per_pix,
                ]
            )

        field_coords_tensor: torch.Tensor = torch.tensor(field_coords, dtype=torch.float32)
        # Extract field coordinates and add batch dimension
        field_x = field_coords_tensor[:, 0].unsqueeze(1).to(self.device_val)[None, ...]
        field_y = field_coords_tensor[:, 1].unsqueeze(1).to(self.device_val)[None, ...]

        # Vectorized focal and band values - ensure correct shape for forward method
        focal_val = focal.expand(centers.shape[0], 1).to(self.device_val)[None, ...]
        band_val = filter_name.expand(centers.shape[0], 1).to(self.device_val)[None, ...]

        image_tensor = F.to_tensor(image)[None, ...]
        with torch.no_grad():
            pred = self.forward_shifts(image_tensor, field_x, field_y, focal_val, band_val, shift_amount)
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
        filter_name = header["FILTER"]
        if detectorName is None:
            detectorName = header["CHIPID"]
        #  U G R I Z Y
        if "u" in filter_name:
            filter_name = torch.tensor([0])
        elif "g" in filter_name:
            filter_name = torch.tensor([1])
        elif "r" in filter_name:
            filter_name = torch.tensor([2])
        elif "i" in filter_name:
            filter_name = torch.tensor([3])
        elif "z" in filter_name:
            filter_name = torch.tensor([4])
        elif "y" in filter_name:
            filter_name = torch.tensor([5])

        # check if intra or extra
        if header["CCDSLOT"][2:] == "1":
            focal = torch.tensor([0]).float()
        elif header["CCDSLOT"][2:] == "0":
            focal = torch.tensor([1]).float()

        centers = get_centers(image, self.CROP_SIZE)

        # Cache camera transforms to avoid repeated instantiation
        camera_transforms = LsstCameraTransforms(camera)

        # Vectorized field coordinate computation
        field_coords = []
        for x, y in centers:
            coord = camera_transforms.ccdPixelToFocalMm(x, y, detectorName=detectorName)
            field_coords.append(
                [
                    coord[0] / self.mm_pix * self.deg_per_pix,
                    coord[1] / self.mm_pix * self.deg_per_pix,
                ]
            )

        field_coords_tensor: torch.Tensor = torch.tensor(field_coords, dtype=torch.float32)
        # Extract field coordinates and add batch dimension
        field_x = field_coords_tensor[:, 0].unsqueeze(1).to(self.device_val)[None, ...]
        field_y = field_coords_tensor[:, 1].unsqueeze(1).to(self.device_val)[None, ...]

        # Vectorized focal and band values - ensure correct shape for forward_align method
        focal_val = focal.expand(centers.shape[0], 1).to(self.device_val)[None, ...]
        band_val = filter_name.expand(centers.shape[0], 1).to(self.device_val)[None, ...]

        image_tensor = F.to_tensor(image)[None, ...]
        with torch.no_grad():
            aligned_images = self.forward_align(
                image_tensor.to(self.device_val), field_x, field_y, focal_val, band_val
            )
        return aligned_images
