"""Utility for plotting the learned donut kernel weights."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Optional


def plot_donut_kernel_weights(
    model,
    save_path: Optional[str] = None,
    figsize: tuple = (20, 12),
    dpi: int = 150,
    show_plot: bool = True
):
    """
    Plot the learned convolution kernel weights and dynamic mask generation.
    
    Parameters
    ----------
    model : WaveNetSystem
        The trained model with dynamic donut kernel.
    save_path : Optional[str], default=None
        Path to save the plot. If None, plot is not saved.
    figsize : tuple, default=(20, 12)
        Figure size for the plot.
    dpi : int, default=150
        DPI for the saved plot.
    show_plot : bool, default=True
        Whether to display the plot.
    """
    
    # Get the convolution weights
    conv_weights = model.get_donut_kernel_weights()
    
    # Convert to numpy arrays
    conv_weights_np = conv_weights.squeeze().numpy()  # Remove batch and channel dims
    
    # Create the plot
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    
    # Plot 1: Convolution kernel weights (2D heatmap)
    im1 = axes[0, 0].imshow(conv_weights_np, cmap='viridis', interpolation='nearest')
    axes[0, 0].set_title('3x3 Convolution Kernel', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Kernel X')
    axes[0, 0].set_ylabel('Kernel Y')
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
    
    # Plot 2: Convolution kernel as 3D surface
    ax3d = fig.add_subplot(3, 3, 2, projection='3d')
    x, y = np.meshgrid(range(conv_weights_np.shape[0]), range(conv_weights_np.shape[1]))
    surf = ax3d.plot_surface(x, y, conv_weights_np, cmap='viridis', alpha=0.8)
    ax3d.set_title('Convolution Kernel 3D', fontsize=14, fontweight='bold')
    ax3d.set_xlabel('Kernel X')
    ax3d.set_ylabel('Kernel Y')
    ax3d.set_zlabel('Weight Value')
    
    # Plot 3: Center cross-sections
    center_x = conv_weights_np.shape[0] // 2
    center_y = conv_weights_np.shape[1] // 2
    axes[0, 2].plot(conv_weights_np[center_x, :], 'b-', linewidth=2, label='Horizontal')
    axes[0, 2].plot(conv_weights_np[:, center_y], 'r-', linewidth=2, label='Vertical')
    axes[0, 2].set_title('Center Cross-Sections', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('Kernel Position')
    axes[0, 2].set_ylabel('Weight Value')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Weight distribution
    axes[1, 0].hist(conv_weights_np.flatten(), bins=50, alpha=0.7, color='green')
    axes[1, 0].set_title('Convolution Weight Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Weight Value')
    axes[1, 0].set_ylabel('Frequency')
    
    # Plot 5: Kernel statistics
    axes[1, 1].bar(['Mean', 'Std', 'Min', 'Max'], 
                   [conv_weights_np.mean(), conv_weights_np.std(), 
                    conv_weights_np.min(), conv_weights_np.max()],
                   color=['blue', 'green', 'red', 'orange'], alpha=0.7)
    axes[1, 1].set_title('Kernel Statistics', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Value')
    
    # Plot 6: U-Net architecture info
    unet_info = """
    Dynamic Mask Generator (U-Net):
    - Encoder: 32→64→128→256→512 channels
    - Decoder: 512→256→128→64→32 channels
    - Skip connections: Preserve fine details
    - Output: Sigmoid activation (0-1)
    - Temperature: 0.5 (sharper masks)
    - Parameters: ~1.2M
    """
    axes[1, 2].text(0.1, 0.5, unet_info, fontsize=10, 
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    axes[1, 2].set_title('U-Net Architecture', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    
    # Plot 7: Dynamic mask example (placeholder)
    axes[2, 0].text(0.5, 0.5, 'Dynamic Mask\n(Generated per image)', 
                     ha='center', va='center', fontsize=12, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    axes[2, 0].set_title('Dynamic Mask Example', fontsize=14, fontweight='bold')
    axes[2, 0].axis('off')
    
    # Plot 8: Mask statistics (placeholder)
    axes[2, 1].text(0.5, 0.5, 'Mask Statistics\n(Per image basis)', 
                     ha='center', va='center', fontsize=12, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    axes[2, 1].set_title('Mask Statistics', fontsize=14, fontweight='bold')
    axes[2, 1].axis('off')
    
    # Plot 9: Architecture comparison
    comparison_text = """
    Architecture Comparison:
    
    Fixed Mask (Old):
    - Static binary mask
    - Same for all images
    - No learning
    
    Dynamic Mask (New):
    - U-Net generated mask
    - Adapts to each image
    - Learns optimal regions
    """
    axes[2, 2].text(0.1, 0.5, comparison_text, fontsize=9, 
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    axes[2, 2].set_title('Architecture Comparison', fontsize=14, fontweight='bold')
    axes[2, 2].axis('off')
    
    # Add statistics
    fig.suptitle('Dynamic Donut Kernel Analysis (U-Net + Convolution)', fontsize=16, fontweight='bold')
    
    # Add text with statistics
    stats_text = f"""
    Convolution Kernel Statistics:
    
    Kernel Shape: {conv_weights_np.shape}
    - Range: [{conv_weights_np.min():.3f}, {conv_weights_np.max():.3f}]
    - Mean: {conv_weights_np.mean():.3f}
    - Std: {conv_weights_np.std():.3f}
    - Sum: {conv_weights_np.sum():.3f}
    - Center Value: {conv_weights_np[center_x, center_y]:.3f}
    
    U-Net Mask Generator:
    - Total Parameters: ~1.2M
    - Dynamic masks per image
    - Temperature scaling: 0.5
    - Output range: [0, 1]
    """
    
    # Add text box
    fig.text(0.02, 0.02, stats_text, fontsize=9, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    # Show plot
    if show_plot:
        plt.show()
    
    return fig


def plot_dynamic_masks(
    model,
    sample_batch: dict,
    num_samples: int = 4,
    save_path: Optional[str] = None,
    figsize: tuple = (20, 15),
    dpi: int = 150,
    show_plot: bool = True
):
    """
    Plot dynamic masks generated by the U-Net for different input images.
    
    Parameters
    ----------
    model : WaveNetSystem
        The trained model with dynamic donut kernel.
    sample_batch : dict
        A batch of data containing images.
    num_samples : int, default=4
        Number of sample images to plot.
    save_path : Optional[str], default=None
        Path to save the plot. If None, plot is not saved.
    figsize : tuple, default=(20, 15)
        Figure size for the plot.
    dpi : int, default=150
        DPI for the saved plot.
    show_plot : bool, default=True
        Whether to display the plot.
    """
    
    # Get images from the batch
    images = sample_batch["image"]
    
    # Limit to num_samples
    num_samples = min(num_samples, images.shape[0])
    images = images[:num_samples]
    
    # Generate masks for each sample
    masks = []
    for i in range(num_samples):
        img = images[i]  # Get single image (no batch dimension)
        
        # Handle multi-channel input by taking the first channel
        if img.dim() == 3 and img.shape[0] > 1:
            img = img[0:1, :, :]  # Take only the first channel for 3D tensor
        elif img.dim() == 4 and img.shape[1] > 1:
            img = img[:, 0:1, :, :]  # Take only the first channel for 4D tensor
        
        # Add batch dimension if needed
        if img.dim() == 3:
            img = img.unsqueeze(0)  # Add batch dimension
        
        # Get the raw mask from U-Net (before temperature scaling)
        mask = model.donut_kernel.mask_generator.get_mask(img)
        masks.append(mask.squeeze().detach().cpu().numpy())
    
    # Convert images to numpy
    images_np = images.detach().cpu().numpy()
    
    # Create the plot
    fig, axes = plt.subplots(num_samples, 4, figsize=figsize)
    
    # If only one sample, ensure axes is 2D
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Original image
        im1 = axes[i, 0].imshow(images_np[i], cmap='viridis', interpolation='nearest')
        if i == 0:
            axes[i, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[i, 0].set_ylabel(f'Sample {i+1}')
        axes[i, 0].set_xlabel('X')
        axes[i, 0].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[i, 0], shrink=0.8)
        
        # Generated mask (raw)
        im2 = axes[i, 1].imshow(masks[i], cmap='viridis', interpolation='nearest')
        if i == 0:
            axes[i, 1].set_title('Generated Mask (Raw)', fontsize=14, fontweight='bold')
        axes[i, 1].set_xlabel('X')
        axes[i, 1].set_ylabel('Y')
        plt.colorbar(im2, ax=axes[i, 1], shrink=0.8)
        
        # Mask with temperature scaling
        temp_mask = torch.sigmoid(torch.logit(torch.tensor(masks[i])) / 0.5).numpy()
        im3 = axes[i, 2].imshow(temp_mask, cmap='viridis', interpolation='nearest')
        if i == 0:
            axes[i, 2].set_title('Mask (Temperature=0.5)', fontsize=14, fontweight='bold')
        axes[i, 2].set_xlabel('X')
        axes[i, 2].set_ylabel('Y')
        plt.colorbar(im3, ax=axes[i, 2], shrink=0.8)
        
        # Binary mask (thresholded)
        binary_mask = (temp_mask > 0.5).astype(float)
        im4 = axes[i, 3].imshow(binary_mask, cmap='gray', interpolation='nearest')
        if i == 0:
            axes[i, 3].set_title('Binary Mask (>0.5)', fontsize=14, fontweight='bold')
        axes[i, 3].set_xlabel('X')
        axes[i, 3].set_ylabel('Y')
        plt.colorbar(im4, ax=axes[i, 3], shrink=0.8)
    
    # Add statistics
    fig.suptitle('Dynamic Mask Generation by U-Net', fontsize=16, fontweight='bold')
    
    # Calculate overall statistics
    all_masks = np.concatenate([mask.flatten() for mask in masks])
    all_temp_masks = np.concatenate([torch.sigmoid(torch.logit(torch.tensor(mask)) / 0.5).numpy().flatten() for mask in masks])
    
    stats_text = f"""
    Dynamic Mask Statistics:
    
    Raw Masks:
    - Range: [{all_masks.min():.3f}, {all_masks.max():.3f}]
    - Mean: {all_masks.mean():.3f}
    - Std: {all_masks.std():.3f}
    
    Temperature Scaled (0.5):
    - Range: [{all_temp_masks.min():.3f}, {all_temp_masks.max():.3f}]
    - Mean: {all_temp_masks.mean():.3f}
    - Std: {all_temp_masks.std():.3f}
    
    Sample Images: {num_samples}
    """
    
    # Add text box
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Dynamic mask plot saved to: {save_path}")
    
    # Show plot
    if show_plot:
        plt.show()
    
    return fig


def plot_image_transformation(
    model,
    sample_batch: dict,
    num_samples: int = 4,
    save_path: Optional[str] = None,
    figsize: tuple = (20, 12),
    dpi: int = 150,
    show_plot: bool = True
):
    """
    Plot images before and after donut kernel processing.
    
    Parameters
    ----------
    model : WaveNetSystem
        The trained model with donut kernel.
    sample_batch : dict
        A batch of data containing images.
    num_samples : int, default=4
        Number of sample images to plot.
    save_path : Optional[str], default=None
        Path to save the plot. If None, plot is not saved.
    figsize : tuple, default=(20, 12)
        Figure size for the plot.
    dpi : int, default=150
        DPI for the saved plot.
    show_plot : bool, default=True
        Whether to display the plot.
    """
    
    # Get images from the batch
    images = sample_batch["image"]
    
    # Limit to num_samples
    num_samples = min(num_samples, images.shape[0])
    images = images[:num_samples]
    
    # Process images with donut kernel
    processed_images = []
    original_images = []
    
    for i in range(num_samples):
        img = images[i:i+1]  # Keep batch dimension
        img_before, img_after = model.process_image_with_donut_kernel(img)
        original_images.append(img_before.squeeze().detach().cpu())
        processed_images.append(img_after.squeeze().detach().cpu())
    
    # Convert to numpy arrays
    original_images = [img.numpy() for img in original_images]
    processed_images = [img.numpy() for img in processed_images]
    
    # Create the plot
    fig, axes = plt.subplots(num_samples, 3, figsize=figsize)
    
    # If only one sample, ensure axes is 2D
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Original image
        im1 = axes[i, 0].imshow(original_images[i], cmap='viridis', interpolation='nearest')
        if i == 0:
            axes[i, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[i, 0].set_ylabel(f'Sample {i+1}')
        axes[i, 0].set_xlabel('X')
        axes[i, 0].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[i, 0], shrink=0.8)
        
        # Processed image
        im2 = axes[i, 1].imshow(processed_images[i], cmap='viridis', interpolation='nearest')
        if i == 0:
            axes[i, 1].set_title('After Donut Kernel', fontsize=14, fontweight='bold')
        axes[i, 1].set_xlabel('X')
        axes[i, 1].set_ylabel('Y')
        plt.colorbar(im2, ax=axes[i, 1], shrink=0.8)
        
        # Difference
        diff = processed_images[i] - original_images[i]
        im3 = axes[i, 2].imshow(diff, cmap='RdBu_r', interpolation='nearest')
        if i == 0:
            axes[i, 2].set_title('Difference (After - Before)', fontsize=14, fontweight='bold')
        axes[i, 2].set_xlabel('X')
        axes[i, 2].set_ylabel('Y')
        plt.colorbar(im3, ax=axes[i, 2], shrink=0.8)
    
    # Add statistics
    fig.suptitle('Image Transformation by Donut Kernel', fontsize=16, fontweight='bold')
    
    # Calculate overall statistics
    all_original = np.concatenate([img.flatten() for img in original_images])
    all_processed = np.concatenate([img.flatten() for img in processed_images])
    all_diff = all_processed - all_original
    
    stats_text = f"""
    Overall Statistics:
    
    Original Images:
    - Range: [{all_original.min():.3f}, {all_original.max():.3f}]
    - Mean: {all_original.mean():.3f}
    - Std: {all_original.std():.3f}
    
    Processed Images:
    - Range: [{all_processed.min():.3f}, {all_processed.max():.3f}]
    - Mean: {all_processed.mean():.3f}
    - Std: {all_processed.std():.3f}
    
    Difference:
    - Range: [{all_diff.min():.3f}, {all_diff.max():.3f}]
    - Mean: {all_diff.mean():.3f}
    - Std: {all_diff.std():.3f}
    
    Sample Images: {num_samples}
    """
    
    # Add text box
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Image transformation plot saved to: {save_path}")
    
    # Show plot
    if show_plot:
        plt.show()
    
    return fig


def plot_donut_kernel_comparison(
    model_before,
    model_after,
    save_path: Optional[str] = None,
    figsize: tuple = (24, 12),
    dpi: int = 150,
    show_plot: bool = True
):
    """
    Plot comparison of donut kernel weights before and after training with softmax normalization.
    
    Parameters
    ----------
    model_before : WaveNetSystem
        The model before training (initial weights).
    model_after : WaveNetSystem
        The model after training (learned weights).
    save_path : Optional[str], default=None
        Path to save the plot. If None, plot is not saved.
    figsize : tuple, default=(24, 12)
        Figure size for the plot.
    dpi : int, default=150
        DPI for the saved plot.
    show_plot : bool, default=True
        Whether to display the plot.
    """
    
    # Get weights from both models
    weights_before = model_before.get_donut_kernel_masked_weights().numpy()
    weights_after = model_after.get_donut_kernel_masked_weights().numpy()
    
    # Calculate normalized weights for both models
    def get_normalized_weights(model):
        weights = model.get_donut_kernel_weights()
        mask = model.get_donut_kernel_mask()
        masked_weights = weights * mask
        masked_weights_flat = masked_weights.view(-1)
        normalized_weights = torch.softmax(masked_weights_flat, dim=0)
        return normalized_weights.view(masked_weights.shape).numpy()
    
    normalized_before = get_normalized_weights(model_before)
    normalized_after = get_normalized_weights(model_after)
    
    # Create the plot
    fig, axes = plt.subplots(3, 4, figsize=figsize)
    
    # Row 1: Before training
    im1 = axes[0, 0].imshow(weights_before, cmap='viridis', interpolation='nearest')
    axes[0, 0].set_title('Before Training (Raw)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Before Training')
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
    
    im2 = axes[0, 1].imshow(normalized_before, cmap='viridis', interpolation='nearest')
    axes[0, 1].set_title('Before Training (Normalized)', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
    
    axes[0, 2].hist(weights_before.flatten(), bins=50, alpha=0.7, color='blue')
    axes[0, 2].set_title('Before Training Distribution', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Weight Value')
    axes[0, 2].set_ylabel('Frequency')
    
    axes[0, 3].bar(['Raw', 'Normalized'], 
                   [weights_before.sum(), normalized_before.sum()],
                   color=['blue', 'red'], alpha=0.7)
    axes[0, 3].set_title('Before Training Sum', fontsize=12, fontweight='bold')
    axes[0, 3].set_ylabel('Sum')
    
    # Row 2: After training
    im3 = axes[1, 0].imshow(weights_after, cmap='viridis', interpolation='nearest')
    axes[1, 0].set_title('After Training (Raw)', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('After Training')
    plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)
    
    im4 = axes[1, 1].imshow(normalized_after, cmap='viridis', interpolation='nearest')
    axes[1, 1].set_title('After Training (Normalized)', fontsize=12, fontweight='bold')
    plt.colorbar(im4, ax=axes[1, 1], shrink=0.8)
    
    axes[1, 2].hist(weights_after.flatten(), bins=50, alpha=0.7, color='green')
    axes[1, 2].set_title('After Training Distribution', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Weight Value')
    axes[1, 2].set_ylabel('Frequency')
    
    axes[1, 3].bar(['Raw', 'Normalized'], 
                   [weights_after.sum(), normalized_after.sum()],
                   color=['green', 'orange'], alpha=0.7)
    axes[1, 3].set_title('After Training Sum', fontsize=12, fontweight='bold')
    axes[1, 3].set_ylabel('Sum')
    
    # Row 3: Differences
    diff_raw = weights_after - weights_before
    diff_normalized = normalized_after - normalized_before
    
    im5 = axes[2, 0].imshow(diff_raw, cmap='RdBu_r', interpolation='nearest')
    axes[2, 0].set_title('Raw Difference', fontsize=12, fontweight='bold')
    axes[2, 0].set_ylabel('Difference')
    plt.colorbar(im5, ax=axes[2, 0], shrink=0.8)
    
    im6 = axes[2, 1].imshow(diff_normalized, cmap='RdBu_r', interpolation='nearest')
    axes[2, 1].set_title('Normalized Difference', fontsize=12, fontweight='bold')
    plt.colorbar(im6, ax=axes[2, 1], shrink=0.8)
    
    axes[2, 2].hist(diff_raw.flatten(), bins=50, alpha=0.7, color='purple')
    axes[2, 2].set_title('Raw Difference Distribution', fontsize=12, fontweight='bold')
    axes[2, 2].set_xlabel('Weight Difference')
    axes[2, 2].set_ylabel('Frequency')
    
    axes[2, 3].hist(diff_normalized.flatten(), bins=50, alpha=0.7, color='brown')
    axes[2, 3].set_title('Normalized Difference Distribution', fontsize=12, fontweight='bold')
    axes[2, 3].set_xlabel('Weight Difference')
    axes[2, 3].set_ylabel('Frequency')
    
    # Add labels
    for ax in axes[0, :]:
        ax.set_xlabel('X')
    
    # Statistics table
    stats_text = f"""
    Statistics:
    
    Raw Weights:
    - Before: mean={weights_before.mean():.4f}, std={weights_before.std():.4f}
    - After: mean={weights_after.mean():.4f}, std={weights_after.std():.4f}
    - Diff: mean={diff_raw.mean():.4f}, std={diff_raw.std():.4f}
    
    Normalized Weights:
    - Before: mean={normalized_before.mean():.4f}, std={normalized_before.std():.4f}
    - After: mean={normalized_after.mean():.4f}, std={normalized_after.std():.4f}
    - Diff: mean={diff_normalized.mean():.4f}, std={diff_normalized.std():.4f}
    
    Sums:
    - Raw Before: {weights_before.sum():.4f}
    - Raw After: {weights_after.sum():.4f}
    - Norm Before: {normalized_before.sum():.4f}
    - Norm After: {normalized_after.sum():.4f}
    """
    
    # Add text box
    fig.text(0.02, 0.02, stats_text, fontsize=8, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    fig.suptitle('Donut Kernel Training Comparison with Softmax Normalization', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    # Show plot
    if show_plot:
        plt.show()
    
    return fig


if __name__ == "__main__":
    # Example usage
    print("This module provides plotting utilities for donut kernel weights.")
    print("Import and use plot_donut_kernel_weights() or plot_donut_kernel_comparison()") 