import os
import time
import torch
import lightning as L
import torch.nn.functional as F
from box import Box
from torch.utils.data import DataLoader
from lightning.fabric.fabric import _FabricOptimizer
from lightning.fabric.loggers import TensorBoardLogger
import segmentation_models_pytorch as smp
from .utils import (
    AverageMeter,
    Metrics,
    validate,
    print_and_log_metrics,
    plot_history,
    save
)
from .losses import (
    DiceLoss,
    FocalLoss,
    CalcIoU,
    CalcDSC
)
from ..model import FinestSAM
from .utils import configure_opt
from ..dataset import load_dataset


def call_train(cfg: Box):
    # Set up the output directory
    main_directory = os.path.dirname(os.path.abspath(__file__)).rsplit('/', 2)[0]
    cfg.sav_dir = os.path.join(main_directory, cfg.sav_dir)
    cfg.out_dir = os.path.join(main_directory, cfg.out_dir)

    loggers = [TensorBoardLogger(cfg.sav_dir, name="loggers_finestSAM")]

    fabric = L.Fabric(accelerator=cfg.device,
                      devices=cfg.num_devices,
                      strategy="auto",
                      num_nodes=cfg.num_nodes, 
                      loggers=loggers)
    
    fabric.launch(train, cfg)


def train(fabric, *args, **kwargs):
    """
    Main training function.
    
    Args:
        fabric (L.Fabric): The lightning fabric.
        *args: The positional arguments:
            [0] - cfg (Box): The configuration file.
        **kwargs: The keyword arguments:
            not used.
    """
    # Get the arguments
    cfg = args[0]

    fabric.seed_everything(cfg.seed_device)

    if fabric.global_rank == 0: 
        os.makedirs(os.path.join(cfg.sav_dir, "loggers_finestSAM"), exist_ok=True)

    with fabric.device:
        model = FinestSAM(cfg)
        model.setup()
        model.train()
        model.to(fabric.device)

    # Load the dataset
    train_data, val_data = load_dataset(cfg, model.model.image_encoder.img_size)
    train_data = fabric._setup_dataloader(train_data)
    val_data = fabric._setup_dataloader(val_data)

    # Configure the optimizer and scheduler
    optimizer, scheduler = configure_opt(cfg, model)
    model, optimizer = fabric.setup(model, optimizer)

    train_loop(cfg, fabric, model, optimizer, scheduler, train_data, val_data) 

def train_loop(
    cfg: Box,
    fabric: L.Fabric,
    model: FinestSAM,
    optimizer: _FabricOptimizer,
    scheduler: _FabricOptimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
):
    """
    The SAM training loop.
    """

    # Initialize the losses
    focal_loss = FocalLoss(gamma=cfg.losses.focal_gamma, alpha=cfg.losses.focal_alpha)
    dice_loss = DiceLoss()
    # uncommented this lines if you want use them instead of smp.metrics
    # calc_iou = CalcIoU()
    # calc_dsc = CalcDSC()

    if cfg.prompts.use_logits: cfg.prompts.use_masks = False
    epoch_logits = []

    last_lr = scheduler.get_last_lr()
    best_val_iou = 0.
    best_val_dsc = 0.
    best_iou_ckpt_path = ""
    best_dsc_ckpt_path = ""

    plots = os.path.join(cfg.out_dir, "plots")
    os.makedirs(plots, exist_ok=True)
    metrics_history = {
        "total_loss": [],
        "focal_loss": [],
        "dice_loss": [],
        "iou_loss": [],
        "train_iou": [],
        "train_dsc": [],
        "val_iou": [],
        "val_dsc": [],
        "epochs": [],
    }

    for epoch in range(1, cfg.num_epochs+1):
        # Initialize the meters
        epoch_metrics = Metrics()
        end = time.time()

        for iter, batched_data in enumerate(train_dataloader):
            torch.cuda.empty_cache()
            epoch_metrics.data_time.update(time.time()-end)

            if epoch > 1 and cfg.prompts.use_logits: [data.update({"mask_inputs": logits.clone().detach().unsqueeze(1)}) for data, logits in zip(batched_data, epoch_logits)]

            outputs = model(batched_input=batched_data, multimask_output=cfg.multimask_output, are_logits=cfg.prompts.use_logits)

            batched_pred_masks = []
            batched_iou_predictions = []
            batched_logits = []
            for item in outputs:
                batched_pred_masks.append(item["masks"])
                batched_iou_predictions.append(item["iou_predictions"])
                batched_logits.append(item["low_res_logits"])

            batch_size = len(batched_data)

            iter_metrics = {
                "loss_focal": torch.tensor(0., device=fabric.device),
                "loss_dice": torch.tensor(0., device=fabric.device),
                "loss_iou": torch.tensor(0., device=fabric.device),
                "iou": torch.tensor(0., device=fabric.device),
                "iou_pred": torch.tensor(0., device=fabric.device),
                "dsc": torch.tensor(0., device=fabric.device),
            }

            # Compute the losses
            for data, pred_masks, iou_predictions, logits in zip(batched_data, batched_pred_masks, batched_iou_predictions, batched_logits):

                if cfg.multimask_output:
                    separated_masks = torch.unbind(pred_masks, dim=1)
                    separated_scores = torch.unbind(iou_predictions, dim=1)
                    separated_logits = torch.unbind(logits, dim=1)

                    best_index = torch.argmax(torch.tensor([torch.mean(score) for score in separated_scores]))
                    pred_masks = separated_masks[best_index]
                    iou_predictions = separated_scores[best_index]
                    logits = separated_logits[best_index]
                else:
                    pred_masks = pred_masks.squeeze(1)
                    iou_predictions = iou_predictions.squeeze(1)
                    logits = logits.squeeze(1)

                if cfg.prompts.use_logits: epoch_logits.append(logits)

                batch_stats = smp.metrics.get_stats(
                    pred_masks,
                    data["gt_masks"].int(),
                    mode='binary',
                    threshold=0.5,
                )

                # Update the metrics
                batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
                batch_dsc = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
                # uncommented these lines if you want use them instead of smp.metrics
                # batch_iou = calc_iou(pred_masks, data["gt_masks"])
                # batch_dsc = calc_dsc(pred_masks, data["gt_masks"])
                batch_iou_predictions = torch.mean(iou_predictions)
                
                iter_metrics["iou"] += batch_iou
                iter_metrics["dsc"] += batch_dsc
                iter_metrics["iou_pred"] += batch_iou_predictions

                # Calculate the losses
                iter_metrics["loss_focal"] += focal_loss(pred_masks, data["gt_masks"], len(pred_masks)) 
                iter_metrics["loss_dice"] += dice_loss(pred_masks, data["gt_masks"], len(pred_masks))
                iter_metrics["loss_iou"] += F.mse_loss(batch_iou_predictions, batch_iou, reduction='mean')

            loss_total = cfg.losses.focal_ratio * iter_metrics["loss_focal"] + cfg.losses.dice_ratio * iter_metrics["loss_dice"] + cfg.losses.iou_ratio * iter_metrics["loss_iou"]

            # Backward pass
            optimizer.zero_grad()
            fabric.backward(loss_total)
            optimizer.step()

            epoch_metrics.batch_time.update(time.time() - end)
            end = time.time()

            # Update the meters
            epoch_metrics.focal_losses.update(iter_metrics["loss_focal"].item(), batch_size)
            epoch_metrics.dice_losses.update(iter_metrics["loss_dice"].item(), batch_size)
            epoch_metrics.space_iou_losses.update(iter_metrics["loss_iou"].item(), batch_size)
            epoch_metrics.total_losses.update(loss_total.item(), batch_size)
            epoch_metrics.ious.update(iter_metrics["iou"].item()/batch_size, batch_size)
            epoch_metrics.ious_pred.update(iter_metrics["iou_pred"].item()/batch_size, batch_size)
            epoch_metrics.dsc.update(iter_metrics["dsc"].item()/batch_size, batch_size)

            print_and_log_metrics(fabric, cfg, epoch, iter, epoch_metrics, train_dataloader)

        # Step the scheduler
        scheduler.step(epoch_metrics.total_losses.avg)
        if scheduler.get_last_lr() != last_lr:
            last_lr = scheduler.get_last_lr()
            fabric.print(f"learning rate changed to: {last_lr}")

        if (cfg.eval_interval > 0 and epoch % cfg.eval_interval == 0) or (epoch == cfg.num_epochs):
            
            val_iou, val_dsc = validate(fabric, cfg, model, val_dataloader, epoch)

            # SEPARARE LE METRICHE DI TRAIN E VAL
            # adesso le metriche vengono salvate solo quando viene effettuata una validation ma
            # il grafico di training risulta piu' completo se vengono salvate ad ogni epoca
            # sarebbe da valutare anche a a priori prima di iniziare il training
            metrics_history["epochs"].append(epoch)
            metrics_history["total_loss"].append(epoch_metrics.total_losses.avg)
            metrics_history["focal_loss"].append(cfg.losses.focal_ratio * epoch_metrics.focal_losses.avg)
            metrics_history["dice_loss"].append(cfg.losses.dice_ratio * epoch_metrics.dice_losses.avg)
            metrics_history["iou_loss"].append(cfg.losses.iou_ratio * epoch_metrics.space_iou_losses.avg)
            metrics_history["train_iou"].append(epoch_metrics.ious.avg)
            metrics_history["train_dsc"].append(epoch_metrics.dsc.avg)
            metrics_history["val_iou"].append(val_iou)
            metrics_history["val_dsc"].append(val_dsc)

            plot_history(metrics_history, plots)

            if val_iou > best_val_iou:
                best_val_iou = val_iou
                if os.path.exists(best_iou_ckpt_path):
                    try:
                        os.remove(best_iou_ckpt_path)
                    except OSError as e:
                        fabric.print(f"Error deleting old best_iou checkpoint: {e}")
                
                ckpt_name = f"best_iou_epoch_{epoch}_val_{val_iou:.4f}"
                best_iou_ckpt_path = os.path.join(cfg.sav_dir, ckpt_name + ".pth")
                save(fabric, model, cfg.sav_dir, ckpt_name)
                fabric.print(f"New best IoU model saved: {ckpt_name}.pth")

            if val_dsc > best_val_dsc:
                best_val_dsc = val_dsc
                if os.path.exists(best_dsc_ckpt_path):
                    try:
                        os.remove(best_dsc_ckpt_path)
                    except OSError as e:
                        fabric.print(f"Error deleting old best_dsc checkpoint: {e}")
                
                ckpt_name = f"best_dsc_epoch_{epoch}_val_{val_dsc:.4f}"
                best_dsc_ckpt_path = os.path.join(cfg.sav_dir, ckpt_name + ".pth")
                save(fabric, model, cfg.sav_dir, ckpt_name)
                fabric.print(f"New best DSC model saved: {ckpt_name}.pth")