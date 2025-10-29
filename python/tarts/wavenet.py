"""Neural network to predict zernike coefficients from donut images and positions."""

import torch
from torch import nn
from torchvision import models as cnn_models
import timm
from .KERNEL import CUTOUT as DONUT
# import torch.nn.functional as F_  # Unused import

# # Global cache for Gaussian kernels
# _GAUSSIAN_KERNEL_CACHE = {}

# def get_gaussian_kernel2d(kernel_size: int, sigma: float, device=None, dtype=torch.float32) -> torch.Tensor:
#     """
#     Returns a 2D Gaussian kernel as a [1, 1, k, k] tensor for depthwise conv.
#     Uses caching to avoid recomputation.

#     Parameters:
#     - kernel_size (int): Must be odd.
#     - sigma (float): Standard deviation of Gaussian.

#     Returns:
#     - kernel (torch.Tensor): Shape [1, 1, k, k].
#     """
#     if kernel_size % 2 == 0:
#         raise ValueError("Kernel size must be odd")

#     # Create cache key
#     cache_key = (kernel_size, sigma, device, dtype)

#     if cache_key in _GAUSSIAN_KERNEL_CACHE:
#         return _GAUSSIAN_KERNEL_CACHE[cache_key]

#     ax = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
#     xx, yy = torch.meshgrid(ax, ax, indexing='ij')
#     kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
#     kernel /= kernel.sum()
#     kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, k, k]

#     # Cache the kernel
#     _GAUSSIAN_KERNEL_CACHE[cache_key] = kernel
#     return kernel

# def apply_gaussian_blur(image: torch.Tensor, kernel_size: int, sigma: float, donut_mask=None) -> torch.Tensor:
#     """
#     Apply 2D Gaussian blur to a 4D PyTorch tensor [B, C, H, W].
#     Optimized version with kernel caching and reduced memory operations.

#     Parameters:
#     - image: torch.Tensor - Input image tensor [B, C, H, W]
#     - kernel_size: int - Size of Gaussian kernel
#     - sigma: float - Standard deviation of Gaussian
#     - donut_mask: torch.Tensor, optional - Pre-computed donut mask to avoid repeated computation

#     Returns:
#     - torch.Tensor: Blurred image of same shape as input
#     """
#     if image.dim() != 4:
#         raise ValueError("Input must be a 4D tensor [B, C, H, W]")

#     B, C, H, W = image.shape
#     device = image.device
#     dtype = image.dtype

#     # Get cached kernel
#     kernel = get_gaussian_kernel2d(kernel_size, sigma, device, dtype)  # [1, 1, k, k]
#     kernel = kernel.repeat(C, 1, 1, 1)  # [C, 1, k, k] for depthwise conv

#     # Pad to keep image size
#     padding = kernel_size // 2

#     # Use provided donut mask or compute it
#     if donut_mask is None:
#         donut = DONUT[40:200, 40:200].float().to(device)
#     else:
#         donut = donut_mask

#     # Optimize the operations: combine operations to reduce memory usage
#     # Instead of: back = image - donut * image
#     # Use: back = image * (1 - donut)
#     back = image * (1 - donut)

#     # Apply blur
#     blurred = F_.conv2d(image, kernel, padding=padding, groups=C)

#     # Combine operations: blurred = donut * blurred + back
#     # This reduces one memory allocation
#     result = donut * blurred + back

#     return result


class WaveNet(nn.Module):
    """Transfer learning driven WaveNet."""

    def __init__(
        self,
        cnn_model: str = "resnet18",
        freeze_cnn: bool = False,
        n_predictor_layers: tuple = (256,),
        n_zernikes: int = 25,
        device='cuda',
        pretrained: bool = True,
    ) -> None:
        """Create the WaveNet.

        Parameters
        ----------
        cnn_model: str, default="resnet18"
            The name of the pre-trained CNN model from torchvision or timm.
            Supports both torchvision models (e.g., "resnet18") and timm models (e.g., "mobilenetv4_conv_small").
        freeze_cnn: bool, default=False
            Whether to freeze the CNN weights.
        n_predictor_layers: tuple, default=(256)
            Number of nodes in the hidden layers of the Zernike predictor network.
            This does not include the output layer, which is determined by n_zernikes.
        n_zernikes: int, default=25
            Number of Zernike coefficients to predict.
        device: str, default='cuda'
            Device to run the model on ('cuda' or 'cpu').
        pretrained: bool, default=True
            Whether to use pre-trained CNN weights. Set to False to avoid downloading weights.
        """
        super().__init__()
        # Check CUDA availability and convert device string to torch.device
        if isinstance(device, str):
            self.device_val = torch.device(device if torch.cuda.is_available() else "cpu")
        else:
            self.device_val = device if torch.cuda.is_available() else torch.device("cpu")

        # load the CNN
        self._load_cnn_backbone(cnn_model, pretrained)

        # remove the final fully connected layer
        self._remove_final_layer()

        if freeze_cnn:
            # freeze cnn parameters
            self.cnn.eval()
            for param in self.cnn.parameters():
                param.requires_grad = False

        # create linear layers that predict zernikes
        n_meta_features = 4  # includes field_x, field_y, intra flag, wavelen
        n_features = self.n_cnn_features + n_meta_features

        if len(n_predictor_layers) > 0:
            # start with the very first layer
            layers = [
                nn.Linear(n_features, n_predictor_layers[0]),
                nn.BatchNorm1d(n_predictor_layers[0]),
                nn.ReLU(),
                nn.Dropout(p=0.2)
            ]

            # add any additional layers
            for i in range(1, len(n_predictor_layers)):
                layers += [
                    nn.Linear(n_predictor_layers[i - 1], n_predictor_layers[i]),
                    nn.BatchNorm1d(n_predictor_layers[i]),
                    nn.ReLU()
                ]

            # add the final layer
            layers += [nn.Linear(n_predictor_layers[-1], n_zernikes)]

        else:
            layers = [nn.Linear(n_features, n_zernikes)]

        self.predictor = nn.Sequential(*layers).to(self.device_val)

        # Cache the donut mask to avoid repeated computation
        self._donut_mask = None

        # Ensure all model parameters are in float32 to avoid dtype mismatches
        self.float()

    def _load_cnn_backbone(self, cnn_model: str, pretrained: bool) -> None:
        """Load CNN backbone from either torchvision or timm.

        Parameters
        ----------
        cnn_model : str
            The name of the CNN model.
        pretrained : bool
            Whether to use pre-trained weights.
        """
        # Check if it's a timm model (MobileNetV4, etc.)
        if cnn_model.startswith("mobilenetv4") or cnn_model in timm.list_models():
            # Load from timm
            self.cnn = timm.create_model(
                cnn_model,
                pretrained=pretrained,
                num_classes=0  # This removes the classifier and returns pooled features
            ).to(self.device_val).float()  # Explicitly convert to float32
            self.is_timm_model = True

            # Get actual feature dimension by doing a dummy forward pass
            # This is more reliable than trusting num_features for some models
            self.cnn.eval()  # Set to eval mode to avoid batch norm issues with batch_size=1
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224).to(self.device_val)
                dummy_output = self.cnn(dummy_input)
                # Handle both 2D (batch, features) and 4D (batch, channels, h, w) outputs
                if len(dummy_output.shape) == 4:
                    dummy_output = torch.nn.functional.adaptive_avg_pool2d(dummy_output, (1, 1))
                    dummy_output = dummy_output.flatten(1)
                self.n_cnn_features = dummy_output.shape[1]
            self.cnn.train()  # Set back to training mode
        else:
            # Load from torchvision
            weights_param = "DEFAULT" if pretrained else None
            self.cnn = getattr(cnn_models, cnn_model)(weights=weights_param).to(self.device_val).float()  # Explicitly convert to float32
            # Get feature dimension
            self.n_cnn_features = self.cnn.fc.in_features
            self.is_timm_model = False

    def _remove_final_layer(self) -> None:
        """Remove the final fully connected layer from the CNN."""
        if self.is_timm_model:
            # For timm models, we already set num_classes=0, so nothing to do
            pass
        else:
            # For torchvision models, remove the final fully connected layer
            if hasattr(self.cnn, 'fc'):
                self.cnn.fc = nn.Identity()

    def _get_donut_mask(self, device):
        """Get cached donut mask."""
        if self._donut_mask is None or self._donut_mask.device != device:
            self._donut_mask = DONUT[40:200, 40:200].float().to(device)
        return self._donut_mask

    def _reshape_image(self, image: torch.Tensor) -> torch.Tensor:
        """Add 3 identical channels to image tensor."""
        # add a channel dimension
        image = image[..., None, :, :]

        # Get the number of input channels required by the CNN
        if hasattr(self.cnn, 'conv1'):
            # torchvision models (ResNet, etc.)
            n_channels = self.cnn.conv1.in_channels
        elif hasattr(self.cnn, 'conv_stem'):
            # timm models (MobileNet, etc.)
            n_channels = self.cnn.conv_stem.in_channels
        else:
            # Default to 3 channels if we can't determine
            n_channels = 3

        # duplicate image for each channel
        image = image.repeat_interleave(n_channels, dim=-3)

        return image

    def forward(
        self,
        image: torch.Tensor,
        fx: torch.Tensor,
        fy: torch.Tensor,
        intra: torch.Tensor,
        band: torch.Tensor,
    ) -> torch.Tensor:
        """Predict Zernikes from donut image, location, and wavelength.

        Parameters
        ----------
        image: torch.Tensor
            The donut image
        fx: torch.Tensor
            X angle of source with respect to optic axis (radians)
        fy: torch.Tensor
            Y angle of source with respect to optic axis (radians)
        intra: torch.Tensor
            Boolean indicating whether the donut is intra or extra focal
        band: torch.Tensor
            Float or integer indicating which band the donut was observed in.

        Returns
        -------
        torch.Tensor
            Array of Zernike coefficients (Noll indices 4-23; microns)
        """
        # reshape the image
        image = self._reshape_image(image)

        # Get cached donut mask for this device (for potential future use)
        # donut_mask = self._get_donut_mask(image.device)

        # use cnn to extract image features with optimized blur
        # image = apply_gaussian_blur(image.clone(), kernel_size=21, sigma=15, donut_mask=donut_mask)
        cnn_features = self.cnn(image.to(self.device_val))

        # For timm models with num_classes=0, we might still get 4D feature maps
        # depending on the model architecture. Apply pooling if needed.
        if len(cnn_features.shape) == 4:
            # cnn_features shape: [batch, channels, height, width]
            # Apply global average pooling to get [batch, channels]
            cnn_features = torch.nn.functional.adaptive_avg_pool2d(cnn_features, (1, 1))
            cnn_features = cnn_features.flatten(1)

        # Store features for inspection/debugging (clone to avoid affecting forward pass)
        # self.cnn_features = cnn_features.clone().detach().cpu()
        # predict zernikes from all features
        features = torch.cat(
            [
                cnn_features,
                fx.to(self.device_val),
                fy.to(self.device_val),
                intra.to(self.device_val),
                band.to(self.device_val),
            ],
            dim=1,
        )
        # Penultimate features (output of predictor just before final linear layer)
        if len(self.predictor) >= 1:
            pre_logits = self.predictor[:-1](features)
            self.predictor_features = pre_logits.clone().detach().cpu()
            zernikes = self.predictor[-1](pre_logits)
        else:
            # Edge case: no layers (shouldn't happen); fallback
            self.predictor_features = features.clone().detach().cpu()
            zernikes = features

        return zernikes
