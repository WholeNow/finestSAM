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
