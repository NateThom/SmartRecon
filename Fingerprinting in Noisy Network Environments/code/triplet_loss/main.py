import argparse
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
import numpy as np
import matplotlib.pyplot as plt
import wandb

from yaml_config_hook import yaml_config_hook
from MLP import MLP
from dataset import Triplet_Dataset

class TripletLearner(LightningModule):
    def __init__(self, args):
        super().__init__()

        # self.hparams = args
        self.args = args

        # initialize the model
        self.model = MLP(args.in_features, args.out_features, args.num_layers, args.num_nodes_per_layer,
                         activation='ReLU', dropout=0.5)

    def triplet_loss(self, anchor, positive, negative):
        loss = torch.mean((anchor - positive) ** 2)
        return loss

    def forward(self, anchor, positive, negative):
        anchor_rep = self.model(anchor)
        positive_rep = self.model(positive)
        negative_rep = self.model(negative)
        return anchor_rep, positive_rep, negative_rep

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        anchor_in, positive_in, negative_in = batch
        anchor_out, positive_out, negative_out = self.forward(anchor_in, positive_in, negative_in)
        loss = self.triplet_loss(anchor_out, positive_out, negative_out)
        self.log("Training Loss", loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return loss

    def test_step(self, test_batch, batch_idx):
        anchor_in, positive_in, negative_in = test_batch
        anchor_out, positive_out, negative_out = self.forward(anchor_in, positive_in, negative_in)
        anchor_out = anchor_out.cpu().numpy()

    def configure_optimizers(self):
        if self.args.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        else:
            raise NotImplementedError

        return {"optimizer": optimizer}

if __name__ == "__main__":
    wandb_logger = WandbLogger(project="smart_attacker_unsupervised", entity='unr-mpl')
    config_path = "./config.yaml"

    parser = argparse.ArgumentParser(description="smart_attacker_unsupervised")
    yaml_config = yaml_config_hook(config_path)

    sweep = False
    if sweep:
        hyperparameter_defaults = dict(
            input_size=32,
            num_layers=1,
            num_nodes_per_layer=500,
            output_size=128,
            learning_rate=0.0003
        )

        wandb.init(config=hyperparameter_defaults)

        yaml_config = yaml_config_hook(config_path)
        wandb.config.update(
            {k:v for k, v in yaml_config.items() if k not in wandb.config}
        )

        for k, v in wandb.config.items():
            parser.add_argument(f"--{k}", default=v, type=type(v))

        args = parser.parse_args()
    else:
        for k, v in yaml_config.items():
            parser.add_argument(f"--{k}", default=v, type=type(v))

        args = parser.parse_args()

    pl.seed_everything(args.seed)

    if args.train:
        train_dataset = Triplet_Dataset(
            args,
            fold="training",
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, drop_last=False,
                                  shuffle=True)

    elif args.test:
        if args.dataset == "Triplet":
            test_dataset = Triplet_Dataset(
                args,
                fold="testing",
                transform=None
            )

        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers, drop_last=False)

    if args.reload:
        tl = TripletLearner.load_from_checkpoint(args.model_path + args.model_file, args=args)
    else:
        tl = TripletLearner(args)

    if args.save == True:
        checkpoint_callback = ModelCheckpoint(
            monitor='Training Loss',
            dirpath=args.model_path,
            filename='{epoch:02d}-{Training Loss:.05f}-' + f"{args.num_layers}-" +
                     f"{args.num_nodes_per_layer}-" + f"{args.learning_rate}",
            save_top_k=1,
            mode='min',
        )

        trainer = pl.Trainer(
            logger=wandb_logger,
            precision=16,
            callbacks=[checkpoint_callback],
            accelerator='ddp',
            plugins=DDPPlugin(find_unused_parameters=False),
            gpus=args.gpus,
            num_nodes=1,
            # limit_train_batches=0.01,
            # limit_val_batches=0.3,
            max_epochs=args.epochs
        )
    else:
        trainer = pl.Trainer(
            logger=wandb_logger,
            precision=16,
            checkpoint_callback=False,
            accelerator='ddp',
            plugins=DDPPlugin(find_unused_parameters=False),
            gpus=args.gpus,
            num_nodes=1,
            # limit_train_batches=0.01,
            max_epochs=args.epochs
        )

    if args.train == True:
        trainer.fit(tl, train_loader)

    # if args.train == True:
    #     # trainer.fit(net, train_loader, val_loader)
    #     trainer.fit(tl, train_loader)
    #
    # if args.val_only == True:
    #     trainer.test(tl, val_loader)
    #
    elif args.test == True:
        trainer.test(tl, test_loader)
