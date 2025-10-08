"""Neural network to predict donut placement coefficients from donut images and positions."""

import torch
from torch import nn
from torchvision import models as cnn_models


class AlignNet(nn.Module):
    """NeuralAlignment Network.

    A transfer learning-based model for predicting 2D donut location
    using a pre-trained CNN backbone.

    This model leverages a pre-trained CNN (e.g., ResNet)
    to extract features from input images and
    combines these with additional meta-features
    (such as angles and intra/extra focal status) to predict
    positions

    Parameters
    ----------
    cnn_model : str, optional, default="resnet34"
        The name of the pre-trained CNN model from
        `torchvision.models` to be used as the feature extractor.
    freeze_cnn : bool, optional, default=False
        Whether to freeze the CNN weights, preventing updates during training.
    n_predictor_layers : tuple of int, optional, default=(256,)
        The number of nodes in each hidden layer of the Zernike
        predictor network. The output layer is fixed
        to have 2 nodes, corresponding to the predicted Zernike coefficients.

    Notes
    -----
    - If `freeze_cnn` is `True`, all CNN parameters are frozen,
        and the model operates in evaluation mode.
    - The final output layer predicts two values corresponding to x-y positions.
    """

    def __init__(
        self,
        cnn_model: str = "resnet34",
        freeze_cnn: bool = False,
        n_predictor_layers: tuple = (256,),
        device='cuda',
        pretrained: bool = True
    ) -> None:
        """Initialize the NeuralAlignment model.

        Parameters
        ----------
        cnn_model : str, optional, default="resnet34"
            The name of the pre-trained CNN model from
            `torchvision.models`.
        freeze_cnn : bool, optional, default=False
            Whether to freeze the CNN weights, preventing
            updates during training.
        n_predictor_layers : tuple of int, optional, default=(256,)
            The number of nodes in each hidden layer of
            the Zernike predictor network.
            The output layer is fixed to have 2 nodes.
        device : str, optional, default='cuda'
            The device to use for computations ('cuda' or 'cpu').
        pretrained : bool, optional, default=True
            Whether to use pre-trained CNN weights. Set to False to avoid downloading weights.

        Notes
        -----
        - The CNN backbone is loaded from `torchvision.models`
            and modified to remove the final fully connected layer.
        - If `freeze_cnn` is `True`, all CNN parameters are frozen,
            and the model is set to evaluation mode.
        - The predictor network is a fully connected MLP that takes
            extracted CNN features plus four additional meta-features.
        - The final output layer predicts two values.

        """
        super().__init__()
        self.device_val = device
        # load the CNN
        weights_param = "DEFAULT" if pretrained else None
        self.cnn = getattr(cnn_models, cnn_model)(weights=weights_param).to('cpu')

        # save the number of cnn features
        self.n_cnn_features = self.cnn.fc.in_features

        # remove the final fully connected layer
        self.cnn.fc = nn.Identity()

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
                    nn.ReLU(),
                ]

            # add the final layer
            layers += [nn.Linear(n_predictor_layers[-1], 2)]

        else:
            layers = [nn.Linear(n_features, 2)]

        self.predictor = nn.Sequential(*layers)

    def _reshape_image(self, image: torch.Tensor) -> torch.Tensor:
        """Expand a single-channel image tensor to have three identical channels.

        This is to make the data RESNET friendly.

        Parameters
        ----------
        image : torch.Tensor
            A 2D or 3D image tensor of shape (batch_size, height, width)
            or (height, width). Assumes a single-channel input.

        Returns
        -------
        torch.Tensor
            The reshaped image tensor with three
            identical channels, having shape
            (batch_size, 3, height, width) or
            (3, height, width), depending on the input.

        Notes
        -----
        - A channel dimension is first added to the input tensor.
        - The single-channel image is then duplicated to match
            the number of input channels
          required by the first convolutional layer of the CNN.
        """
        # add a channel dimension
        image = image[..., None, :, :]
        # duplicate image for each channel
        image = image.repeat_interleave(self.cnn.conv1.in_channels, dim=-3)
        return image

    def forward(
        self,
        image: torch.Tensor,
        fx: torch.Tensor,
        fy: torch.Tensor,
        intra: torch.Tensor,
        band: torch.Tensor,
    ) -> torch.Tensor:
        """Predict offset from donut image, location, and wavelength.

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
            Array of donut locations (x,y) where (x) corresponds to
            the *first* entry in a tensor and the (y) corresponds to the
            *second*
        """
        # reshape the image
        image = self._reshape_image(image)
        # use cnn to extract image features
        cnn_features = self.cnn(image.to(self.device_val))

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
        offset = self.predictor(features)
        # offsets
        return offset
