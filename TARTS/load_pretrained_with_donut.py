"""Utility for loading pretrained WaveNet weights while handling new donut kernel."""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from .lightning_wavenet_mask import WaveNetSystem


def load_pretrained_with_donut(
    checkpoint_path: str,
    model: WaveNetSystem,
    strict: bool = False,
    map_location: Optional[str] = None,
    freeze_except_donut: bool = False,
) -> WaveNetSystem:
    """
    Load pretrained weights from a checkpoint, handling the new donut kernel.
    
    This function loads weights for all existing parameters and initializes
    the new donut kernel parameters from scratch.
    
    Parameters
    ----------
    checkpoint_path : str
        Path to the pretrained checkpoint file.
    model : WaveNetSystem
        The model instance to load weights into.
    strict : bool, default=False
        Whether to strictly enforce that the keys in state_dict match
        the keys returned by this module's state_dict() function.
    map_location : Optional[str], default=None
        Specifies the device to load the model on.
    freeze_except_donut : bool, default=False
        If True, freeze all model parameters except the donut kernel parameters.
        This is useful for fine-tuning only the donut kernel while keeping
        pretrained weights fixed.
        
    Returns
    -------
    WaveNetSystem
        The model with loaded weights (existing parameters) and initialized
        donut kernel (new parameters).
        
    Notes
    -----
    - Existing parameters from the pretrained model will be loaded
    - New donut kernel parameters will be initialized from scratch
    - The function handles the mismatch between old and new model architectures
    - If freeze_except_donut=True, only donut kernel parameters will be trainable
    """
    
    # Load the checkpoint
    if map_location is None:
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    # Extract the state dict from the checkpoint
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    
    # Get the current model's state dict
    current_state_dict = model.state_dict()
    
    # Filter state dict to only include keys that exist in the current model
    # and exclude the donut kernel parameters
    filtered_state_dict = {}
    donut_kernel_keys = []
    
    for key, value in state_dict.items():
        if key in current_state_dict:
            # Check if this is a donut kernel parameter
            if "donut_kernel" in key:
                donut_kernel_keys.append(key)
                print(f"Skipping donut kernel parameter: {key}")
            else:
                filtered_state_dict[key] = value
                print(f"Loading parameter: {key}")
        else:
            print(f"Skipping parameter not in current model: {key}")
    
    # Load the filtered state dict
    missing_keys, unexpected_keys = model.load_state_dict(
        filtered_state_dict, strict=False
    )
    
    # Freeze all parameters except donut kernel if requested
    if freeze_except_donut:
        print("\nFreezing all parameters except donut kernel...")
        for name, param in model.named_parameters():
            if "donut_kernel" in name:
                param.requires_grad = True
                print(f"Keeping trainable: {name}")
            else:
                param.requires_grad = False
                print(f"Freezing: {name}")
    
    # Print summary
    print(f"\nLoaded {len(filtered_state_dict)} parameters from checkpoint")
    print(f"Skipped {len(donut_kernel_keys)} donut kernel parameters (initialized from scratch)")
    print(f"Missing keys: {len(missing_keys)}")
    print(f"Unexpected keys: {len(unexpected_keys)}")
    
    if missing_keys:
        print("Missing keys (will be initialized from scratch):")
        for key in missing_keys:
            print(f"  - {key}")
    
    if unexpected_keys:
        print("Unexpected keys (ignored):")
        for key in unexpected_keys:
            print(f"  - {key}")
    
    return model


def create_model_with_pretrained_weights(
    checkpoint_path: str,
    freeze_except_donut: bool = False,
    **model_kwargs
) -> WaveNetSystem:
    """
    Create a WaveNetSystem model and load pretrained weights, handling the donut kernel.
    
    Parameters
    ----------
    checkpoint_path : str
        Path to the pretrained checkpoint file.
    freeze_except_donut : bool, default=False
        If True, freeze all model parameters except the donut kernel parameters.
        This is useful for fine-tuning only the donut kernel while keeping
        pretrained weights fixed.
    **model_kwargs
        Keyword arguments to pass to WaveNetSystem constructor.
        
    Returns
    -------
    WaveNetSystem
        The model with loaded weights and initialized donut kernel.
    """
    
    # Create the model
    model = WaveNetSystem(**model_kwargs)
    
    # Load pretrained weights
    model = load_pretrained_with_donut(
        checkpoint_path, 
        model, 
        freeze_except_donut=freeze_except_donut
    )
    
    return model


def print_model_parameters(model: nn.Module, prefix: str = "") -> None:
    """
    Print all model parameters with their shapes and whether they require gradients.
    
    Parameters
    ----------
    model : nn.Module
        The model to print parameters for.
    prefix : str, default=""
        Prefix for parameter names.
    """
    print(f"\nModel Parameters Summary:")
    print(f"{'Parameter':<50} {'Shape':<20} {'Requires Grad':<15}")
    print("-" * 85)
    
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        param_name = f"{prefix}.{name}" if prefix else name
        shape_str = str(list(param.shape))
        requires_grad = param.requires_grad
        
        print(f"{param_name:<50} {shape_str:<20} {str(requires_grad):<15}")
        
        total_params += param.numel()
        if requires_grad:
            trainable_params += param.numel()
    
    print("-" * 85)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")


if __name__ == "__main__":
    # Example usage
    checkpoint_path = "path/to/your/pretrained/checkpoint.ckpt"
    
    # Create model with pretrained weights
    model = create_model_with_pretrained_weights(
        checkpoint_path,
        cnn_model="resnet34",
        n_predictor_layers=(256,),
        lr=1e-3
    )
    
    # Print model parameters
    print_model_parameters(model) 