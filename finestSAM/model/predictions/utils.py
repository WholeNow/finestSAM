import os
import torch
import numpy as np
from typing import Dict, Union, Optional, Any
import matplotlib.pyplot as plt

def show_anns(
        anns: list, 
        opacity: float = 0.35
    ):
    '''
    Show annotations on the image.

    Args:
        anns (list): The list of annotations, which is the output list of the automatic predictor.
        opacity (float): The opacity of the masks.
    '''

    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [opacity]])
        img[m] = color_mask
    ax.imshow(img)


def show_mask(mask, ax, random_color=True, seed=None):
    '''
    Show a single mask on the image.
    
    Args:
        mask (torch.Tensor): The mask to be shown.
        ax (matplotlib.axes.Axes): The axes to show the mask on.
        random_color (bool): Whether to use a random color for the mask.
        seed (int): The seed for the random color.
    '''
    np.random.seed(seed)
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.cpu().numpy().reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1].cpu().numpy()
    neg_points = coords[labels==0].cpu().numpy()
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   


import matplotlib.pyplot as plt
def show_predictions(
    original_image: np.ndarray, 
    gt_masks: Union[torch.Tensor, np.ndarray], 
    pred_masks: Union[torch.Tensor, np.ndarray], 
    cfg: Optional[Dict[str, Any]] = None, 
    epoch: Optional[int] = None, 
    iter: Optional[int] = None, 
    idx: Optional[int] = None, 
    save: bool = False
    ) -> None:
    '''
    Display side-by-side comparison of original image, ground truth masks, and predicted masks.
    
    Args:
    original_image: The original input image
    gt_masks: Ground truth segmentation masks (torch.Tensor or numpy array)
    pred_masks: Predicted segmentation masks (torch.Tensor or numpy array)
    cfg: Optional configuration dictionary with visualization settings
    epoch: Current epoch (for saving)
    iter: Current iteration (for saving)
    idx: Sample index (for saving)
    save: Whether to save the visualization
    '''
    
    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Set color seed for consistent colors
    color_seed = 42 # You can change this to any integer for different colors
    np.random.seed(color_seed)
    
    # 1. Original image
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    

    # 2. Ground truth masks
    axes[1].imshow(original_image)
    axes[1].set_title("Ground Truth Masks")
    # Show ground truth masks with different colors
    if isinstance(gt_masks, torch.Tensor):
        gt_masks = gt_masks.cpu().detach()
    
    # NOTA: Assumendo che 'show_mask' esista e funzioni come previsto
    # Handle different dimensions for ground truth masks
    if len(gt_masks.shape) == 4:  # [B, C, H, W]
        for i in range(gt_masks.shape[1]):
            show_mask(gt_masks[0, i] > 0, axes[1], random_color=True, seed=i+color_seed)
    elif len(gt_masks.shape) == 3:  # [C, H, W]
        for i in range(gt_masks.shape[0]):
            show_mask(gt_masks[i] > 0, axes[1], random_color=True, seed=i+color_seed)
    else:  # [H, W]
        show_mask(gt_masks > 0, axes[1], random_color=True, seed=color_seed)
    
    axes[1].axis('off')
    

    # 3. Predicted masks
    axes[2].imshow(original_image)
    axes[2].set_title("Predicted Masks")
    
    # Show predicted masks with different colors
    if isinstance(pred_masks, torch.Tensor):
        pred_masks = pred_masks.cpu().detach()
    
    # NOTA: Assumendo che 'show_mask' esista e funzioni come previsto
    # Handle different dimensions for predicted masks
    if len(pred_masks.shape) == 4:  # [B, C, H, W]
        for i in range(pred_masks.shape[1]):
            # Add offset to seed to ensure different colors from ground truth
            show_mask(pred_masks[0, i] > 0, axes[2], random_color=True, seed=i+color_seed)
    elif len(pred_masks.shape) == 3:  # [C, H, W]
        for i in range(pred_masks.shape[0]):
            show_mask(pred_masks[i] > 0, axes[2], random_color=True, seed=i+color_seed)
    else:  # [H, W]
        show_mask(pred_masks > 0, axes[2], random_color=True, seed=color_seed)
    
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save figure if requested
    if save and cfg is not None and hasattr(cfg, 'out_dir'):
        out_vis = os.path.join(cfg.out_dir, "visualizations")
        os.makedirs(out_vis, exist_ok=True)
        vis_path = os.path.join(out_vis, f"epoch{epoch}_iter{iter}_sample{idx}.png")
        plt.savefig(vis_path)
    
    # CORREZIONE: Sostituisci plt.show() con plt.close(fig)
    # Questo previene il blocco dello script e l'output <Figure ...>
    # plt.show() # <-- Rimuovi
    plt.close(fig) # <-- Aggiungi