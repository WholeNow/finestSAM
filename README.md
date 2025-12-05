# finestSAM

This project was carried out as part of the thesis at the University of Cagliari by:


* [`Marco Pilia`](https://github.com/Marchisceddu)
* [`Simone Dessi`](https://github.com/Druimo)

The main goal is to perform fine-tuning of the Segment-Anything model by MetaAI on a custom dataset in COCO format, with the aim of providing an effective implementation for predictions using SAM's automatic predictor.
The code utilizes the Fabric framework from Lightning AI to offer an efficient implementation of the model.

> [!NOTE]
> Currently, this project implements a **classic fine-tuning** approach.

To read the full research conducted for solving the task, you can consult the thesis (in Italian) at [`link`](https://drive.google.com/file/d/1JJwgVJOXWdbUqyN0FSMuvoJxFhoqSF4g/view?usp=sharing)

## Dataset

You can structure your dataset in two ways, depending on whether you want the script to automatically split it into training and validation sets or if you prefer to provide them manually.

### Option 1: Auto-Split
If you want the script to handle the split, organize your folder as follows:

```
dataset/
└── data/
    ├── images/           # Folder containing all images
    │   ├── 0.png
    │   └── ...
    └── annotations.json  # COCO annotations for all images
```

### Option 2: Pre-Split
If you already have separate training and validation sets:

```
dataset/
├── train/
│   ├── images/
│   └── annotations.json
└── val/
    ├── images/
    └── annotations.json
```

## Setup

Here are the steps to follow:

1. **Download the SAM model checkpoint**  
   The instructions for downloading the SAM model checkpoint can be found in the [`finestSAM/sav/`](https://github.com/Marchisceddu/finestSAM/blob/main/finestSAM/sav/) directory.

2. **Install necessary dependencies:**

    - Install dependencies using pip by running the following command from the project directory:
      ```bash
      pip install -r requirements.txt
      ```

    - Alternatively, you can create a Conda environment using the provided `environment.yaml` file:
      ```bash
      conda env create -f environment.yaml
      ```

This will ensure that all required packages and libraries are installed and ready for use.

## Config

The hyperparameters required for the model are specified in [`finestSAM/config.py`](https://github.com/Marchisceddu/finestSAM/blob/main/finestSAM/config.py).

<details>
<summary> <b>Configuration Overview</b> </summary>

### **General**
- `device`: Hardware to run the model ("auto", "gpu", "cpu").
- `num_devices`: Number of devices or "auto".
- `num_nodes`: Number of GPU nodes for distributed training.
- `seed_device`: Seed for device reproducibility (or None).
- `sav_dir`: Output folder for model saves.
- `out_dir`: Output folder for predictions.
- `model`:
    - `type`: Model type ("vit_h", "vit_l", "vit_b").
    - `checkpoint`: Path to the .pth checkpoint file.

### **Training**
- `seed_dataloader`: Seed for dataloader reproducibility (or None).
- `batch_size`: Batch size for images.
- `num_workers`: Number of subprocesses for data loading.
- `num_epochs`: Number of training epochs.
- `eval_interval`: Interval (in epochs) for validation.
- `prompts`:
    - `use_boxes`: Use bounding boxes for training.
    - `use_points`: Use points for training.
    - `use_masks`: Use mask annotations for training.
    - `use_logits`: Use logits from previous epoch.
- `multimask_output`: (Bool) Enable multimask output.
- `opt`:
    - `learning_rate`: Learning rate.
    - `weight_decay`: Weight decay.
- `sched`:
    - `type`: Scheduler type ("ReduceLROnPlateau" or "LambdaLR").
    - `LambdaLR`:
        - `decay_factor`: Learning rate decay factor.
        - `steps`: List of steps for decay.
        - `warmup_steps`: Number of warmup epochs.
    - `ReduceLROnPlateau`:
        - `decay_factor`: Learning rate decay factor.
        - `epoch_patience`: Patience for LR decay.
        - `threshold`: Threshold for measuring the new optimum.
        - `cooldown`: Number of epochs to wait before resuming normal operation.
        - `min_lr`: Minimum learning rate.
- `losses`:
    - `focal_ratio`: Weight of focal loss.
    - `dice_ratio`: Weight of dice loss.
    - `iou_ratio`: Weight of IoU loss.
    - `focal_alpha`: Alpha value for focal loss.
    - `focal_gamma`: Gamma value for focal loss.
- `model_layer`:
    - `freeze`:
        - `image_encoder`: Freeze image encoder.
        - `prompt_encoder`: Freeze prompt encoder.
        - `mask_decoder`: Freeze mask decoder.

### **Dataset**
- `auto_split`: (Bool) Automatically split dataset.
- `seed`: Seed for dataset operations.
- `use_cache`: (Bool) Use cached dataset metadata.
- `sav`: Filename for saving dataset cache.
- `val_size`: (Float) Validation split percentage.
- `positive_points`: Number of positive points per mask.
- `negative_points`: Number of negative points per mask.
- `use_center`: Use the mask center as a key point.
- `snap_to_grid`: Align points to the automatic predictor grid.

### **Prediction**
- `opacity`: Transparency of predicted masks (0.0 - 1.0).

</details>

## Run model

To execute the file [`finestSAM/__main__.py`](https://github.com/Marchisceddu/finestSAM/blob/main/finestSAM/__main__.py), use the following command-line arguments.

> [!TIP]
> Check out the [`notebook.ipynb`](https://github.com/Marchisceddu/finestSAM/blob/main/notebook.ipynb) for the most up-to-date usage examples and easy experimentation.

### **Training the Model:**
Run the training process by specifying the mode and the dataset path:

```bash
python -m finestSAM --mode "train" --dataset "path/to/dataset"
```

### **Automatic Predictions:**
For making predictions, specify the input image path:

```bash
python -m finestSAM --mode "predict" --input "path/to/image.png"
```

Optionally, modify the mask opacity (default 0.9):

```bash
python -m finestSAM --mode "predict" --input "path/to/image.png" --opacity 0.8
```

## Results

The fine-tuning of the model was carried out to perform efficient instance segmentation, specifically for generating polygons that delimit urban areas in PDFs. These PDFs represent urban planning tools that regulate land transformations, such as areas where specific building restrictions apply. 

For this task, a single prompt was used during training: __1 central point per mask aligned with the automatic predictor grid.__ This prompt proved to be the most effective for training and ensuring the proper functioning of SAM's automatic predictor.

### Test sam vit_b
<details>

<summary> Training progress </summary>

![Train sam vit_b](assets/Test-vit_b/Test6.png)
 _Training progress for the sam_vit_b model_

</details>

<details>

<summary> Comparison images </summary>

![Test 6 - Comparison 1](assets/Test-vit_b/Test6_comparison_1.png)
 _`Original Image`_ , _`Ground Truth Masks`_ , _`SAM vit_b_ - _Masks`_ , _`Finetuning - Masks`_ 

![Test 6 - Comparison 2](assets/Test-vit_b/Test6_comparison_2.png)
 _`Original Image`_ , _`Ground Truth Masks`_ , _`SAM vit_b_ - _Masks`_ , _`Finetuning - Masks`_ 

![Test 6 - Comparison 3](assets/Test-vit_b/Test6_comparison_3.png)
 _`Original Image`_ , _`Ground Truth Masks`_ , _`SAM vit_b_ - _Masks`_ , _`Finetuning - Masks`_ 

![Test 6 - Comparison 4](assets/Test-vit_b/Test6_comparison_4.png)
 _`Original Image`_ , _`Ground Truth Masks`_ , _`SAM vit_b_ - _Masks`_ , _`Finetuning - Masks`_ 

</details>


### Test sam vit_h
<details>

<summary> Training progress </summary>

![Train sam vit_h](assets/Test-vit_h/Test8.png)
 _Training progress for the sam_vit_h model_

</details>

<details>

<summary> Comparison images </summary>

![Test 6 - Comparison 1](assets/Test-vit_h/Test8_comparison_1.png)
 _`Original Image`_ , _`Ground Truth Masks`_ , _`SAM vit_h_ - _Masks`_ , _`Finetuning - Masks`_ 

![Test 6 - Comparison 2](assets/Test-vit_h/Test8_comparison_2.png)
 _`Original Image`_ , _`Ground Truth Masks`_ , _`SAM vit_h_ - _Masks`_ , _`Finetuning - Masks`_ 

![Test 6 - Comparison 3](assets/Test-vit_h/Test8_comparison_3.png)
 _`Original Image`_ , _`Ground Truth Masks`_ , _`SAM vit_h_ - _Masks`_ , _`Finetuning - Masks`_ 

![Test 6 - Comparison 4](assets/Test-vit_h/Test8_comparison_4.png)
 _`Original Image`_ , _`Ground Truth Masks`_ , _`SAM vit_h_ - _Masks`_ , _`Finetuning - Masks`_ 

</details>


## To-Do List

- [ ] Added a function to create the bounding boxes for training (suggestion on line 175 [finestSAM/model/dataset.py](https://github.com/Marchisceddu/finestSAM/blob/main/finestSAM/model/dataset.py))

- [ ] Validation method based on SAM automatic predictor

- [ ] Test

- [ ] tpu support

- [ ] Adapter Lora fine-tuning method

## Resources

- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [Lightning AI](https://github.com/Lightning-AI/lightning)
- [lightning-sam](https://github.com/luca-medeiros/lightning-sam)

## License
The model is licensed under the [Apache 2.0 license](https://github.com/Marchisceddu/finestSAM/blob/main/LICENSE.txt).