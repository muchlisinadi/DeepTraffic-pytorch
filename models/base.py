import pytorch_lightning as pl
import torch
import torchmetrics


class LitBase(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        # Train metrics
        self.train_acc = torchmetrics.Accuracy()
        self.train_f1 = torchmetrics.F1Score(num_classes=kwargs["num_classes"])
        self.train_pmacro = torchmetrics.Precision(
            average="macro", num_classes=kwargs["num_classes"]
        )
        self.train_pmicro = torchmetrics.Precision(average="micro")
        self.train_rmacro = torchmetrics.Recall(
            average="macro", num_classes=kwargs["num_classes"]
        )
        self.train_rmicro = torchmetrics.Recall(average="micro")

        # Val metrics
        self.val_acc = torchmetrics.Accuracy()
        self.val_f1 = torchmetrics.F1Score(num_classes=kwargs["num_classes"])
        self.val_pmacro = torchmetrics.Precision(
            average="macro", num_classes=kwargs["num_classes"]
        )
        self.val_pmicro = torchmetrics.Precision(average="micro")
        self.val_rmacro = torchmetrics.Recall(
            average="macro", num_classes=kwargs["num_classes"]
        )
        self.val_rmicro = torchmetrics.Recall(average="micro")

    def loss_function(self, pred, labels):
        raise NotImplementedError

    def logger_metrics(
        self, suffix, pred, labels, position, on_step, on_train=True, is_write=True
    ):

        if on_train:
            accuracy = (
                self.train_acc(pred, labels) if on_step else self.train_acc.compute()
            )
            f1 = self.train_f1(pred, labels) if on_step else self.train_f1.compute()
            p_macro = (
                self.train_pmacro(pred, labels)
                if on_step
                else self.train_pmacro.compute()
            )
            p_micro = (
                self.train_pmicro(pred, labels)
                if on_step
                else self.train_pmicro.compute()
            )
            r_macro = (
                self.train_rmacro(pred, labels)
                if on_step
                else self.train_rmacro.compute()
            )
            r_micro = (
                self.train_rmicro(pred, labels)
                if on_step
                else self.train_rmicro.compute()
            )
        else:
            accuracy = self.val_acc(pred, labels) if on_step else self.val_acc.compute()
            f1 = self.val_f1(pred, labels) if on_step else self.val_f1.compute()
            p_macro = (
                self.val_pmacro(pred, labels) if on_step else self.val_pmacro.compute()
            )
            p_micro = (
                self.val_pmicro(pred, labels) if on_step else self.val_pmicro.compute()
            )
            r_macro = (
                self.val_rmacro(pred, labels) if on_step else self.val_rmacro.compute()
            )
            r_micro = (
                self.val_rmicro(pred, labels) if on_step else self.val_rmicro.compute()
            )

        if is_write:
            self.logger.experiment.add_scalar("Accuracy/" + suffix, accuracy, position)
            self.logger.experiment.add_scalar("F1/" + suffix, f1, position)
            self.logger.experiment.add_scalar("Pmacro/" + suffix, p_macro, position)
            self.logger.experiment.add_scalar("Pmicro/" + suffix, p_micro, position)
            self.logger.experiment.add_scalar("Rmacro/" + suffix, r_macro, position)
            self.logger.experiment.add_scalar("Rmicro/" + suffix, r_micro, position)

    def training_step(self, batch, batch_idx):
        # REQUIRED- run at every batch of training data
        # extracting input and output from the batch
        x, labels = batch

        # forward pass on a batch
        pred = self.forward(x)

        # calculating the loss
        train_loss = self.loss_function(pred, labels)

        self.logger.experiment.add_scalar(
            "Loss/Step-Train", train_loss, self.global_step
        )
        self.logger_metrics("Step-Train", pred, labels, self.global_step, on_step=True)

        return {"loss": train_loss}

    def validation_step(self, val_batch, batch_idx):
        # REQUIRED- run at every batch of training data
        # extracting input and output from the batch
        x, labels = val_batch

        # forward pass on a batch
        pred = self.forward(x)

        # calculating the loss
        val_loss = self.loss_function(pred, labels)

        self.logger.experiment.add_scalar("Loss/Step-Val", val_loss, self.global_step)
        self.logger_metrics(
            "Step-Val",
            pred,
            labels,
            self.global_step,
            on_step=True,
            on_train=False,
            is_write=False,
        )

        self.log(
            "val_loss",
            val_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return {"loss": val_loss}

    def training_epoch_end(self, outputs):
        #  the function is called after every epoch is completed

        # calculating average loss
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.logger.experiment.add_scalar(
            "Loss/Epoch-Train", avg_loss, self.current_epoch
        )
        self.logger_metrics(
            "Epoch-Train", None, None, self.current_epoch, on_step=False
        )

    def validation_epoch_end(self, outputs) -> None:
        #  the function is called after every epoch is completed

        # calculating average loss
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.logger.experiment.add_scalar(
            "Loss/Epoch-Val", avg_loss, self.current_epoch
        )
        self.logger_metrics(
            "Epoch-Val", None, None, self.current_epoch, on_step=False, on_train=False
        )
