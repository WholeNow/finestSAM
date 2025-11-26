from box import Box

config = {
    "device": "auto",
    "num_devices": "auto",
    "num_nodes": 1,
    "seed_device": 1337,
    "sav_dir": "sav",
    "out_dir": "out",

    "model": {
        "type": 'vit_b',
        "checkpoint": "sam_vit_b_01ec64.pth",
    },
}

config_train = {
    "seed_dataloader": None,
    "batch_size": 1,
    "num_workers": 0,

    "num_epochs": 500,
    "eval_interval": 3,
    "prompts": {
        "use_boxes": False,
        "use_points": True,
        "use_masks": False,
        "use_logits": False,
    },
    "multimask_output": False,

    "opt": {
        "learning_rate": 4e-5,
        "weight_decay": 1e-4,
    },

    "sched": {
        "type": "ReduceLROnPlateau",
        "LambdaLR": {
            "decay_factor": 10, # 1 / (cfg.sched.LambdaLR.decay_factor ** (mul_factor+1))
            "steps": None,
            "warmup_steps": 0,
        },
        "ReduceLROnPlateau": {
            "decay_factor": 0.05, # lr * factor -> 8e-4 * 0.1 = 8e-5
            "epoch_patience": 10,
            "threshold": 1e-4,
            "cooldown": 0,
            "min_lr": 0,
        },
    },

    "losses": {
        "focal_ratio": 20.,
        "dice_ratio": 1.,
        "iou_ratio": 1.,
        "focal_alpha": 0.8,
        "focal_gamma": 2,
    },

    "model_layer": {
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": False,
        },
    },

    "dataset": {
        "auto_split": True,
        "seed": 42,
        "use_cache": True,
        "split_path": {
            "root_dir": "../dataset4img",
            "images_dir": "images",
            "annotation_file": "annotations.json",
            "sav": "sav.pth",
            "val_size": 0.3,
        },
        "no_split_path": {
            "train": {
                "root_dir": "../dataset/train",
                "images_dir": "images",
                "annotation_file": "annotations.json",
                "sav": "sav.pth"
            },
            "val": {
                "root_dir": "../dataset/val",
                "images_dir": "images",
                "annotation_file": "annotations.json",
                "sav": "sav.pth"
            },
        },
        "positive_points": 1,
        "negative_points": 0,
        "use_center": True, # The first positive point is always the most significant for each mask
        "snap_to_grid": True, # Align the center to the prediction grid used by the automatic predictor
    }
}

config_predict = {
    "opacity": 0.9,
}

cfg_train = Box(config)
cfg_train.update(Box(config_train))

cfg_predict = Box(config)
cfg_predict.update(Box(config_predict))