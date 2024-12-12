"""
Author: Mohammad Dehghani Ashkezari <mdehghan@uw.edu>

Exposes pytorch vision model.
"""

import os
from typing import Optional
import numpy as np
import pandas as pd
import joblib
from tqdm.auto import tqdm
import logging
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from sklearn.metrics import r2_score
import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
from torchinfo import summary

import mmVision.settings as st
from mmVision.model.efficient_b0 import Efficient_b0
from mmVision.model.vit import VIT
from mmVision.utils import dataset_utils

logger = logging.getLogger("root_logger")
matplotlib.use('Agg')
matplotlib.rc('font', family="Arial")
matplotlib.rcParams["agg.path.chunksize"] = 10000


class Vision():
    def __init__(self,
                 train_dir: str,
                 test_dir: str,
                 epochs: int,
                 batch_size: int,
                 color_channel: int,
                 image_size: tuple[int, int],
                 learning_rate: float,
                 valid_dir: Optional[str] = None
                 ) -> None:
        """

        Parameters
        -------------
        train_dir: str
            Path to the training dataset.
        test_dir: str
            Path to the test dataset.
        valid_dir: Optional[str], default None
            Path to the validation dataset, if exists.
        epochs: int
            Number of epochs.
        batch_size: int
            Number of images in the dataloader's batch.
        color_channel: int
            Number of color channels.
        image_size: tuple[int, int]
            Image dimensions (height, width). The input images will be resized to these dimensions. 
        learning_rate: float
            Learning rate.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Pytroch model initiated with device: {self.device}")
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.valid_dir = valid_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.color_channel = color_channel
        self.image_size = image_size
        self.learning_rate = learning_rate

        class_df, _ = dataset_utils.classes(self.train_dir)
        # model_wrapper = Efficient_b0(n_classes=len(class_df))
        model_wrapper = VIT(n_classes=len(class_df))
        self.model = model_wrapper.model
        self.model_name = model_wrapper.model_name
        transformer = model_wrapper.transformer

        self.train_dataset = datasets.ImageFolder(root=self.train_dir,
                                                  transform=transformer
                                                  )
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           num_workers=os.cpu_count(),
                                           pin_memory=True
                                           )
        self.test_dataset = datasets.ImageFolder(root=self.test_dir,
                                                 transform=transformer
                                                 )
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=self.batch_size,
                                          shuffle=False,
                                          num_workers=os.cpu_count(),
                                          pin_memory=True
                                          )
        if self.valid_dir:
            self.valid_dataset = datasets.ImageFolder(root=self.valid_dir,
                                                      transform=transformer
                                                      )
            self.valid_dataloader = DataLoader(self.valid_dataset,
                                               batch_size=self.batch_size,
                                               shuffle=True,
                                               num_workers=os.cpu_count(),
                                               pin_memory=True
                                               )

        self.model = self.model.to(self.device)
        summary(self.model,
                input_size=[self.batch_size, self.color_channel, *self.image_size],
                col_names=["input_size", "output_size", "num_params", "trainable"],
                col_width=20,
                row_settings=["var_names"]
                )
        return

    def fit(self, dir_path=st.FIG_DIR) -> dict:
        self.model = self.model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate)


        history = {"epoch": [],
                   "train_loss": [],
                   "train_acc": [],
                   "test_loss": [],
                   "test_acc": []
                   }

        for epoch in range(self.epochs):
            self.model.train()
            train_loss, train_acc = 0, 0
            for batch, (X, y) in enumerate(tqdm(self.train_dataloader)):
                X, y = X.to(self.device), y.to(self.device)
                y_pred = self.model(X)
                loss = self.loss_fn(y_pred, y)
                train_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
                train_acc += (y_pred_class == y).sum().item() / len(y_pred)
            train_loss = train_loss / len(self.train_dataloader)
            train_acc = train_acc / len(self.train_dataloader)

            self.model.eval()
            test_loss, test_acc = 0, 0
            with torch.inference_mode():
                for batch, (X, y) in enumerate(self.test_dataloader):
                    X, y = X.to(self.device), y.to(self.device)
                    test_pred_logits = self.model(X)
                    loss = self.loss_fn(test_pred_logits, y)
                    test_loss += loss.item()
                    test_pred_labels = test_pred_logits.argmax(dim=1)
                    test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))
                test_loss = test_loss / len(self.test_dataloader)
                test_acc = test_acc / len(self.test_dataloader)

            history["epoch"].append(epoch+1)
            history["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
            history["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
            history["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
            history["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)
            logger.info(f"Epoch: {epoch+1}/{self.epochs}, "
                        f"train loss: {history['train_loss'][-1]:.4f}, "
                        f"train accuracy: {history['train_acc'][-1]:.4f}, "
                        f"test loss: {history['test_loss'][-1]:.4f}, "
                        f"test accuracy: {history['test_acc'][-1]:.4f}"
                        )

        self.loss_curve_plot(history=history, dir_path=dir_path)
        return history

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict using the input x.
        """
        x = x.to(self.device)
        with torch.inference_mode():
            y_pred = self.model(x)
        return y_pred

    def loss_curve_plot(self, history: dict, dir_path: str):
        """
        Plots the model's (tensorflow) training history.

        Parameters
        ------------
        history: dict
            loss metrics history stored in a dictionary.
        dir_path: str
            Path to a directory to store the plot.
        """
        os.makedirs(dir_path, exist_ok=True)
        plt.clf()
        plt.figure(figsize=(20, 10))
        # plt.plot(history["epoch"],
        #          np.array(torch.tensor(history["train_loss"]).cpu().numpy()),
        #          "-o", markeredgewidth=0, markersize=6,
        #          color="tomato", alpha=0.7, label="Training Loss"
        #          )
        plt.plot(history["epoch"],
                 history["train_loss"],
                 "-o", markeredgewidth=0, markersize=6,
                 color="tomato", alpha=0.7, label="Training Loss"
                 )
        plt.plot(history["epoch"],
                 history["test_loss"],
                 "-o", markeredgewidth=0, markersize=6,
                 color="dodgerblue", alpha=0.7, label="Test Loss"
                 )
        plt.title(f"Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(fontsize=18)
        plt.savefig(f"{dir_path}{self.model_name}_loss_curve.png", dpi=300)
        plt.close()


        plt.clf()
        plt.figure(figsize=(20, 10))
        plt.plot(history["epoch"],
                 history["train_acc"],
                 "-o", markeredgewidth=0, markersize=6,
                 color="tomato", alpha=0.7, label="Training Accuracy"
                 )
        plt.plot(history["epoch"],
                 history["test_acc"],
                 "-o", markeredgewidth=0, markersize=6,
                 color="dodgerblue", alpha=0.7, label="Test Accuracy"
                 )
        plt.title(f"Accuracy Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend(fontsize=18)
        plt.savefig(f"{dir_path}{self.model_name}_accuracy_curve.png", dpi=300)
        plt.close()
        return
    
    def save_model(self, dir_path: str = st.MODEL_DIR):
        """
        Save the trained model and scaler on local disk.

        Parameters
        ------------
        dir_path: str
            Path to the directory where the model is stored.
        """
        os.makedirs(dir_path, exist_ok=True)
        model_fname = f"{dir_path}{self.model_name}_torch_model.pth"
        torch.save(self.model.state_dict(), model_fname)
        return
