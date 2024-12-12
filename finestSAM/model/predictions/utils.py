import numpy as np
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