import torch
import torch.nn.functional as F
import pytorch_lightning as L
from torchmetrics import Accuracy, ConfusionMatrix
import timm
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

class TimmClassifier(L.LightningModule):
    def __init__(
        self,
        model_name: str = 'resnet18',
        num_classes: int = 2,
        pretrained: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        scheduler_factor: float = 0.1,
        scheduler_patience: int = 10,
        min_lr: float = 1e-6
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = timm.create_model(self.hparams.model_name, pretrained=self.hparams.pretrained, num_classes=self.hparams.num_classes)

        self.train_acc = Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.hparams.num_classes)

        self.train_cm = ConfusionMatrix(task="multiclass", num_classes=self.hparams.num_classes)
        self.val_cm = ConfusionMatrix(task="multiclass", num_classes=self.hparams.num_classes)
        self.test_cm = ConfusionMatrix(task="multiclass", num_classes=self.hparams.num_classes)

        # Create plots directory in the same folder as your logs
        self.plots_dir = Path("plots")
        self.plots_dir.mkdir(exist_ok=True)
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        # Calculate accuracy for this batch
        acc = (preds == y).float().mean()
        
        # Log both step-level and epoch-level metrics
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(x))
        self.log("train/acc", acc, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(x))
        
        # Still update the training accuracy metric for confusion matrix
        self.train_acc(preds, y)
        self.train_cm(preds, y)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        # Calculate accuracy for this batch
        acc = (preds == y).float().mean()
        
        # Log both step-level and epoch-level metrics
        self.log("val/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(x))
        self.log("val/acc", acc, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(x))
        
        # Still update the validation accuracy metric for confusion matrix
        self.val_acc(preds, y)
        self.val_cm(preds, y)
        
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        # Calculate accuracy for this batch
        acc = (preds == y).float().mean()
        
        # Log both step-level and epoch-level metrics
        self.log("test/loss", loss, prog_bar=True, batch_size=len(x))
        self.log("test/acc", acc, prog_bar=True, batch_size=len(x))
        
        # Still update the test accuracy metric for confusion matrix
        self.test_acc(preds, y)
        self.test_cm(preds, y)
        
        return loss

    def _log_confusion_matrix(self, prefix, cm):
        # Only save plot if it's the last epoch
        if self.current_epoch == self.trainer.max_epochs - 1:
            # Create figure and plot confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm.cpu().numpy(), 
                annot=True, 
                fmt='d',
                cmap='Blues',
                xticklabels=range(self.hparams.num_classes),
                yticklabels=range(self.hparams.num_classes)
            )
            plt.title(f'{prefix.capitalize()} Confusion Matrix - Final Epoch')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            
            # Save plot
            save_path = self.plots_dir / f"{prefix}_confusion_matrix_final.png"
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        
        # Don't log raw numbers to CSV anymore

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.hparams.scheduler_factor,
            patience=self.hparams.scheduler_patience,
            min_lr=self.hparams.min_lr
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",  # Changed from val_loss to val/loss to match our logging
            },
        }

    def on_train_epoch_end(self):
        cm = self.train_cm.compute()
        self._log_confusion_matrix('train', cm)
        self.train_cm.reset()

    #def on_validation_epoch_end(self):
        #cm = self.val_cm.compute()
        #self._log_confusion_matrix('val', cm)
        #self.val_cm.reset()

    def on_test_epoch_end(self):
        cm = self.test_cm.compute()
        self._log_confusion_matrix('test', cm)
        self.test_cm.reset()
