import abc
import torch
import torchmetrics
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from . import models
from . import typedef


class LitModule(pl.LightningModule):
    def __init__(
            self,
            lr=1e-3,
            min_lr=1e-5,
            lr_factor=.5,
            lr_patience=10,
    ):
        super().__init__()
        self.save_hyperparameters(
            "lr",
            "min_lr",
            "lr_factor",
            "lr_patience",
        )
        self.binary_f1_score = torchmetrics.classification.BinaryF1Score()
        self.model = None

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config):
        pass

    def forward_batch(self, batch):
        return self.forward(batch)

    @staticmethod
    def compute_loss(forward_res, batch):
        forward_res = forward_res.ravel()
        lmd = batch.y.sum() / len(batch.y)
        lmd = (lmd**0.5)
        weights = lmd * torch.ones_like(forward_res)
        weights[batch.y > 0] = (1 - lmd)
        return F.binary_cross_entropy_with_logits(
            forward_res, batch.y.float(), weight=weights)
    
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.hparams.lr_factor,
                patience=self.hparams.lr_patience,
                min_lr=self.hparams.min_lr,
                verbose=True,
            ),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        loss = self.compute_loss(self.forward_batch(train_batch), train_batch)
        self.log('train_loss', loss, batch_size=train_batch.ptr.shape[0] - 1)
        self.logger.experiment.add_scalar("train_loss", loss, self.trainer.global_step)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        loss = self.compute_loss(self.forward_batch(val_batch), val_batch)
        pred = self.model(val_batch).sigmoid()
        f1_score = self.binary_f1_score(pred.ravel(), val_batch.y)
        
        self.log('val_loss', loss, prog_bar=True, sync_dist=True, batch_size=val_batch.ptr.shape[0] - 1)
        self.logger.experiment.add_scalar("val_loss", loss, self.trainer.global_step)

        self.log('val_f1_score', f1_score, prog_bar=True, sync_dist=True, batch_size=val_batch.ptr.shape[0] - 1)
        self.logger.experiment.add_scalar("val_f1_score", f1_score, self.trainer.global_step)

    @classmethod
    @abc.abstractmethod
    def load_from_configurable_checkpoint(cls, path, config):
        pass


class ConvNet(LitModule):

    def __init__(
            self,
            units: int,
            num_layers: int = 1,
            knn_dim: int = 0,
            lr=1e-3,
            min_lr=1e-5,
            lr_factor=.5,
            lr_patience=10,
    ):
        super().__init__(lr, min_lr, lr_factor, lr_patience)
        self.save_hyperparameters(
            "units",
            "knn_dim",
            "num_layers",
        )
        self.model = models.ConvNet(
            units=units,
            knn_dim=knn_dim,
            num_layers=num_layers,
        )

    @classmethod
    def from_config(cls, config):
        return cls(
            units=config["net"]["units"],
            knn_dim=config["net"]["knn_dim"],
            num_layers=config["net"]["num_layers"],
            lr=config["lit_module"]["optimizer"]["lr"],
            min_lr=config["lit_module"]["scheduler"]["min_lr"],
            lr_factor=config["lit_module"]["scheduler"]["factor"],
            lr_patience=config["lit_module"]["scheduler"]["patience"],
        )
    
    @classmethod
    def load_from_configurable_checkpoint(cls, path, config):
        return cls.load_from_checkpoint(
            path,
            units=config["net"]["units"],
            knn_dim=config["net"]["knn_dim"],
            num_layers=config["net"]["num_layers"],
            lr=config["lit_module"]["optimizer"]["lr"],
            min_lr=config["lit_module"]["scheduler"]["min_lr"],
            lr_factor=config["lit_module"]["scheduler"]["factor"],
            lr_patience=config["lit_module"]["scheduler"]["patience"],
        )
    

class Conv2Net(ConvNet):

    def __init__(
        self,
        units: int,
        num_layers: int = 1,
        knn_dim: int = 0,
        lr=1e-3,
        min_lr=1e-5,
        lr_factor=.5,
        lr_patience=10,
    ):
        super().__init__(
            units=units, 
            num_layers=num_layers, 
            knn_dim=knn_dim, 
            lr=lr, 
            min_lr=min_lr, 
            lr_factor=lr_factor, 
            lr_patience=lr_patience,
        )
        self.model = models.Conv2Net(
            units=units,
            knn_dim=knn_dim,
            num_layers=num_layers,
        )
            

class TSPNet(LitModule):

    def __init__(
            self,
            units: int,
            num_layers: int = 1,
            dropout=0.,
            lr=1e-3,
            min_lr=1e-5,
            lr_factor=.5,
            lr_patience=10,
    ):
        super().__init__(lr, min_lr, lr_factor, lr_patience)
        self.save_hyperparameters(
            "units",
            "num_layers",
        )
        self.model = models.TSPNet(
            embedding_dim=units,
            num_layers=num_layers,
            dropout=dropout,
        )

    @classmethod
    def from_config(cls, config):
        return cls(
            units=config["net"]["units"],
            num_layers=config["net"]["num_layers"],
            dropout=config["net"]["dropout"],
            lr=config["lit_module"]["optimizer"]["lr"],
            min_lr=config["lit_module"]["scheduler"]["min_lr"],
            lr_factor=config["lit_module"]["scheduler"]["factor"],
            lr_patience=config["lit_module"]["scheduler"]["patience"],
        )
    
    @classmethod
    def load_from_configurable_checkpoint(cls, path, config):
        return cls.load_from_checkpoint(
            path,
            units=config["net"]["units"],
            num_layers=config["net"]["num_layers"],
            dropout=config["net"]["dropout"],
            lr=config["lit_module"]["optimizer"]["lr"],
            min_lr=config["lit_module"]["scheduler"]["min_lr"],
            lr_factor=config["lit_module"]["scheduler"]["factor"],
            lr_patience=config["lit_module"]["scheduler"]["patience"],
        )
    

def get(name):
    return {
        "conv_net": ConvNet,
        "conv2_net": Conv2Net,
        "tsp_net": TSPNet,
    }[name]