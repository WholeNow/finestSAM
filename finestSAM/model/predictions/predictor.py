import os
import cv2
import torch
import lightning as L
import matplotlib.pyplot as plt
from box import Box
from .utils import (
    show_anns,
)
from ..model import FinestSAM


def automatic_predictions(
        cfg: Box, 
        path: str,
        opacity: float = 0.35
    ):
    """
    Predict the masks of the image and save them in a png file.
    The directory of the output is specified in the configuration file.
    
    Args:
        cfg (Box): The configuration file.
        path (str): The path of the image.
        approx_accuracy (float): The approximation accuracy of the polygons.
        opacity (float): The opacity of the masks in the final png image.
    """
    # Get the paths
    main_directory = os.path.dirname(os.path.abspath(__file__)).rsplit('/', 2)[0]
    cfg.sav_dir = os.path.join(main_directory, cfg.sav_dir)
    cfg.out_dir = os.path.join(main_directory, cfg.out_dir)
    image_path = os.path.join(main_directory, path)

    # Get the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load the model 
    with torch.no_grad():
        fabric = L.Fabric(accelerator=cfg.device, #  "tpu" is not supported in this version
                      devices=1,
                      strategy="auto")
        
        fabric.seed_everything(cfg.seed_device)

        with fabric.device:
            model = FinestSAM(cfg)
            model.setup()
            model.eval()
            model.to(fabric.device)

        # Predict the masks
        predictor = model.get_automatic_predictor()
        masks = predictor.generate(image)
           
    # Create the output directory if it does not exist
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Save the predictions as a .png file
    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=100)  # 6.4 inches * 100 dpi = 640 pixels
    ax.set_position([0, 0, 1, 1])  # [left, bottom, width, height]
    plt.imshow(image)
    show_anns(masks, opacity=opacity)
    plt.axis('off')
    plt.savefig(os.path.join(cfg.out_dir, "output.png"), dpi=100, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.clf()

    print("Predictions saved in:", os.path.join(cfg.out_dir, "output.png"))