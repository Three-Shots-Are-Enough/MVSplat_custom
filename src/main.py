import os
from pathlib import Path
import warnings

import hydra
import torch
import wandb
from colorama import Fore
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers.wandb import WandbLogger

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def train(cfg_dict: DictConfig):

    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.4)
        sweep_config = {}
    
    if os.environ.get('WANDB_SWEEP_ID'):
        print("Running as part of a sweep")
        import wandb
        wandb.init()
        sweep_config = dict(wandb.config)
        print("Sweep config:", sweep_config)
    
    if sweep_config:
        near = sweep_config.get('near')
        far = sweep_config.get('far')
        if near is not None:
            print(f"Updating near from {cfg_dict.dataset.near} to {near}")
            OmegaConf.update(cfg_dict, "dataset.near", float(near))
        if far is not None:
            print(f"Updating far from {cfg_dict.dataset.far} to {far}")
            OmegaConf.update(cfg_dict, "dataset.far", float(far))
    
    if not os.environ.get('WANDB_SWEEP_ID') and cfg_dict.wandb.mode != "disabled":
        wandb.init(
            project=cfg_dict.wandb.project,
            entity=cfg_dict.wandb.entity,
            name=cfg_dict.wandb.name,
            config=OmegaConf.to_container(cfg_dict, resolve=True)
        )
    
    print("Final config - near:", cfg_dict.dataset.near)
    print("Final config - far:", cfg_dict.dataset.far)

    print("Config dict after update:", cfg_dict)
    print("Dataset near value:", cfg_dict.dataset.near)
    print("Dataset far value:", cfg_dict.dataset.far)
    
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    # Set up the output directory.
    if cfg_dict.output_dir is None:
        output_dir = Path(
            hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
        )
        print("output_dir is None, using hydra runtime output_dir")
    else:  # for resuming
        output_dir = Path(cfg_dict.output_dir)
        print("output_dir is ", output_dir)
        os.makedirs(output_dir, exist_ok=True)
    print(cyan(f"Saving outputs to {output_dir}."))
    latest_run = output_dir.parents[1] / "latest-run"
    os.system(f"rm {latest_run}")
    os.system(f"ln -s {output_dir} {latest_run}")

    # Set up logging with wandb.
    callbacks = []
    if cfg_dict.wandb.mode != "disabled":
        wandb_extra_kwargs = {}
        if cfg_dict.wandb.id is not None:
            wandb_extra_kwargs.update({'id': cfg_dict.wandb.id,
                                       'resume': "must"})
        logger = WandbLogger(
            entity=cfg_dict.wandb.entity,
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=f"{cfg_dict.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
            tags=cfg_dict.wandb.get("tags", None),
            log_model=False,
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
            **wandb_extra_kwargs,
        )
        callbacks.append(LearningRateMonitor("step", True))

        # On rank != 0, wandb.run is None.
        if wandb.run is not None:
            wandb.run.log_code("src")
    else:
        logger = LocalLogger()

    # Set up checkpointing.
    callbacks.append(
        ModelCheckpoint(
            str(output_dir / "checkpoints"),
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,
            monitor="info/global_step",
            mode="max",  # save the latest k ckpt, can do offline test later
        )
    )
    for cb in callbacks:
        cb.CHECKPOINT_EQUALS_CHAR = '_'

    # Prepare the checkpoint for loading.
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()

    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        logger=logger,
        devices="auto",
        num_nodes=cfg.trainer.num_nodes,
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        enable_progress_bar=cfg.mode == "test",
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
    )
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)

    model_kwargs = {
        "optimizer_cfg": cfg.optimizer,
        "test_cfg": cfg.test,
        "train_cfg": cfg.train,
        "encoder": encoder,
        "encoder_visualizer": encoder_visualizer,
        "decoder": get_decoder(cfg.model.decoder, cfg.dataset),
        "losses": get_losses(cfg.loss),
        "step_tracker": step_tracker,
    }
    if cfg.mode == "train" and checkpoint_path is not None and not cfg.checkpointing.resume:
        # Just load model weights, without optimizer states
        # e.g., fine-tune from the released weights on other datasets
        model_wrapper = ModelWrapper.load_from_checkpoint(
            checkpoint_path, **model_kwargs, strict=True)
        print(cyan(f"Loaded weights from {checkpoint_path}."))
    else:
        model_wrapper = ModelWrapper(**model_kwargs)

    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
    )

    if cfg.mode == "train":
        trainer.fit(model_wrapper, datamodule=data_module, ckpt_path=(
            checkpoint_path if cfg.checkpointing.resume else None))
    else:
        trainer.test(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
        )


if __name__ == "__main__":
    train()