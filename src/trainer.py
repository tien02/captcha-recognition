import torch
import torch.nn as nn
import torch.optim as opt
from lightning.pytorch import LightningModule
from torchmetrics import CharErrorRate, WordErrorRate, MetricCollection, ExtendedEditDistance

from src.utils import decoding_pred

class OCRTrainer(LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()
        self.model = model
        self.lr = learning_rate

        self.val_metrics = MetricCollection([
            CharErrorRate(),
            WordErrorRate(),
            ExtendedEditDistance(),
        ])

        self.test_metrics = MetricCollection([
            CharErrorRate(),
            WordErrorRate(),
            ExtendedEditDistance(),
        ])


    def forward(self, inputs):
        return self.model(inputs)
    

    @staticmethod
    def compute_loss(preds, targets):
        preds_softmax = nn.functional.log_softmax(preds, dim=-1)
        preds_softmax = preds_softmax.permute(1,0,2)

        batch_size = preds_softmax.size(1)
        pred_seq_length = preds_softmax.size(0)
        target_seq_length = targets.size(-1)

        input_lengths = torch.full(size=(batch_size,), fill_value=pred_seq_length, dtype=torch.int32)
        target_lengths = torch.full(size=(batch_size,), fill_value=target_seq_length, dtype=torch.int32)   

        loss = nn.functional.ctc_loss(preds_softmax, targets, input_lengths, target_lengths)
        return loss
    

    @staticmethod
    def decode_logits(preds, targets):
        preds = nn.functional.softmax(preds, dim=-1)
        preds = torch.argmax(preds, dim=-1)

        txt_decoded = decoding_pred(pred_logits=preds, target_logits=targets)
        return txt_decoded

    
    def training_step(self, batch):
        data, targets = batch

        preds = self.model(data)
        loss = self.__class__.compute_loss(preds, targets)

        self.log('loss', loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        data, targets = batch

        preds = self.model(data)

        loss = self.__class__.compute_loss(preds, targets)
        txt_decoded = self.__class__.decode_logits(preds, targets)

        self.val_metrics.update(txt_decoded['prediciton'], txt_decoded['target'])
        self.log('val_loss', loss, on_epoch=True, on_step=True, prog_bar=True)
    

    def on_validation_epoch_end(self):
        val_metrics = self.val_metrics.compute()
        self.log_dict(val_metrics, prog_bar=True)
        self.val_metrics.reset()

    
    def test_step(self, batch, batch_idx):
        data, targets = batch

        preds = self.model(data)

        loss = self.__class__.compute_loss(preds, targets)
        txt_decoded = self.__class__.decode_logits(preds, targets)

        self.test_metrics.update(txt_decoded['prediciton'], txt_decoded['target'])
        self.log('test_loss', loss, on_epoch=True, on_step=True, prog_bar=True)
    

    def on_test_epoch_end(self):
        test_metrics = self.test_metrics.compute()
        self.log_dict(test_metrics)
        self.test_metrics.reset()

    def configure_optimizers(self):
        optimizer = opt.Adam(self.model.parameters(),lr=self.lr, eps=1e-6, weight_decay=0.01,)
        
        scheduler = opt.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr = self.lr, total_steps=self.trainer.estimated_stepping_batches)
        
        return {
            "optimizer": optimizer,
            "scheduler": scheduler,
        }