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
from tqdm import tqdm

from yaml_config_hook import yaml_config_hook
from MLP import MLP
from dataset import Triplet_Dataset

class TripletLearner(LightningModule):
    def __init__(self, args):
        super().__init__()

        # self.hparams = args
        self.args = args

        # initialize the model
        self.model = MLP(self.args.in_features, self.args.out_features, self.args.num_layers_hidden_layers,
                         self.args.num_nodes_per_layer, activation='ReLU', dropout=0.5)

        self.criterion = torch.nn.TripletMarginLoss(margin=1.0, p=2)

        print(self.model)

        self.average_loss = 0

    # def triplet_loss(self, anchor, positive, negative):
    #     loss = torch.max(torch.dist(anchor, positive, p=2) - torch.dist(anchor, negative, p=2) + self.args.alpha, 0)
    #     return loss

    def forward(self, anchor, positive, negative):
        anchor_rep = self.model(anchor)
        positive_rep = self.model(positive)
        negative_rep = self.model(negative)
        return anchor_rep, positive_rep, negative_rep

    def training_step(self, batch, batch_idx):
        anchor_in, anchor_label, positive_in, negative_in = batch
        anchor_out, positive_out, negative_out = self.forward(anchor_in, positive_in, negative_in)
        loss = self.criterion(anchor_out, positive_out, negative_out)

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

    sweep = True
    if sweep:
        hyperparameter_defaults = dict(
            input_size=32,
            num_layers=1,
            num_nodes_per_layer=512,
            output_size=23,
            learning_rate=0.0001
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

        # for item in tqdm(train_loader):
        #     continue

    elif args.test:
        test_dataset = Triplet_Dataset(
            args,
            fold="testing",
        )

        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers, drop_last=False,
                                  shuffle=False)

    if args.reload:
        tl = TripletLearner.load_from_checkpoint(args.model_path + args.model_file, args=args)
    else:
        tl = TripletLearner(args)

    if args.save == True:
        checkpoint_callback = ModelCheckpoint(
            monitor='Training Loss',
            dirpath=args.model_path,
            filename='{epoch:02d}-{Training Loss:.05f}-' + f"{args.learning_rate}-" + f"{args.epochs}-" +
                     f"{args.num_hidden_layers}-" + f"{args.num_nodes_per_layer}",
            save_top_k=1,
            mode='min',
        )

        trainer = pl.Trainer(
            logger=wandb_logger,
            precision=16,
            callbacks=[checkpoint_callback],
            accelerator='ddp',
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
