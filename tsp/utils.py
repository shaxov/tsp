import os
import yaml
import numpy as np
import pickle as pkl
from tqdm import tqdm

from . import litmodules

import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning import loggers
from pytorch_lightning import callbacks
from pytorch_lightning.strategies.ddp import DDPStrategy


def _make_train_val_loaders(dataset, loaders_config):
    total_len = len(dataset)
    train_len = int((1 - loaders_config["test_size"]) * total_len)
    val_len = total_len - train_len

    generator = torch.Generator().manual_seed(loaders_config["seed"])
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=generator)

    train_loader = DataLoader(
        train_set,
        batch_size=loaders_config["batch_size"],
        shuffle=True,
        num_workers=loaders_config["num_workers"],
    )
    val_loader = DataLoader(
        val_set,
        batch_size=loaders_config["batch_size"],
        shuffle=False,
        num_workers=loaders_config["num_workers"],
    )
    return train_loader, val_loader


def make_checkpoint_callback(checkpointer_config, model_name):
    return callbacks.ModelCheckpoint(
        dirpath=os.path.join(
            checkpointer_config["path"],
            model_name,
        ),
        save_top_k=checkpointer_config["save_top_k"],
        verbose=checkpointer_config["verbose"],
        monitor=checkpointer_config["monitor"],
        mode=checkpointer_config["mode"],
        save_last=checkpointer_config["save_last"],
    )


def load_model(path_to_checkpoint_folder, checkpoint_name):
    with open(os.path.join(path_to_checkpoint_folder, "config.pkl"), 'rb') as f:
        model_config = pkl.load(f)
        model_name = model_config["lit_module"]["name"]
        cls = litmodules.get(model_name)
        path_to_model = os.path.join(path_to_checkpoint_folder, checkpoint_name)
        model = cls.load_from_configurable_checkpoint(path_to_model, model_config)
    return model


def make_train_val_loaders(path_to_training_config, dataset):
    with open(path_to_training_config, "r") as f:
        loaders_config = yaml.safe_load(f)["loaders"]
    return _make_train_val_loaders(dataset, loaders_config)


def make_model(
        path_to_training_config,
        knn_dim, 
        model_name=""):
    # Training config loading
    with open(path_to_training_config, "r") as f:
        training_config = yaml.safe_load(f)
        lit_module_config = training_config["lit_module"]
        trainer_config = training_config["trainer"]

    # Module config loading and model initialization
    with open(lit_module_config["path"], "r") as f:
        net_config = {
            "knn_dim": knn_dim,
        }
        net_config.update(yaml.safe_load(f))
        name = lit_module_config["name"]
        cls = litmodules.get(name)
        model_config = {
            "net": net_config,
            "lit_module": lit_module_config,
        }
        model = cls.from_config(model_config)

    path_to_model_config = os.path.join(trainer_config["checkpointer"]["path"], model_name)
    os.makedirs(path_to_model_config, exist_ok=True)
    with open(os.path.join(path_to_model_config, "config.pkl"), "wb") as f:
        pkl.dump(model_config, f)
    return model


def make_trainer(path_to_training_config, name, return_chkp_clbk=False):
    with open(path_to_training_config, "r") as f:
        trainer_config = yaml.safe_load(f)["trainer"]

    # TensorBoard logger initialization
    logger = loggers.TensorBoardLogger(trainer_config["logger"]["path"], name=name)
    # Callback initialization for saving module during training
    checkpoint_callback = make_checkpoint_callback(trainer_config["checkpointer"], name)
    # Callback initialization for printing the progress
    progress_bar_config = trainer_config["progress_bar"]
    tqdm_callback = callbacks.TQDMProgressBar(refresh_rate=progress_bar_config["refresh_rate"])
    # Trainer initialization
    trainer_args = dict(
        accelerator=trainer_config["accelerator"],
        max_epochs=trainer_config["max_epochs"],
        logger=logger,
        callbacks=[
            checkpoint_callback,
            tqdm_callback,
        ],
    )
    if trainer_args["accelerator"] != "cpu":
        trainer_args.update(dict(devices=trainer_config["devices"]))
    if len(trainer_config["devices"]) > 1:
        trainer_args.update(dict(strategy=DDPStrategy(find_unused_parameters=False)))
    trainer = Trainer(**trainer_args)
    if return_chkp_clbk:
        return trainer, checkpoint_callback
    return trainer
