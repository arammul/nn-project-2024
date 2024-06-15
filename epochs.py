import sys
import torch
from tqdm import tqdm as tqdm
from collections import defaultdict

from segmentation_models_pytorch.utils.train import Epoch, TrainEpoch, ValidEpoch
from segmentation_models_pytorch.utils.meter import AverageValueMeter


class MultiHeadEpoch(Epoch):
    def run(self, dataloader):
        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {
            metric.__name__: AverageValueMeter() for metric in self.metrics
        }

        with tqdm(
            dataloader,
            desc=self.stage_name,
            file=sys.stdout,
            disable=not (self.verbose),
        ) as iterator:
            for batch in iterator:
                for dataset_name, mini_batch in batch.items():
                    batch[dataset_name]["x"] = mini_batch["x"].to(self.device)
                    batch[dataset_name]["y"] = mini_batch["y"].to(self.device)
                
                loss, predictions = self.batch_update(batch)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    for dataset_name, y_pred in predictions.items():
                        y_true = batch[dataset_name]["y"]
                        metric_value = metric_fn(y_pred, y_true).cpu().detach().numpy()
                        metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)
        return logs


class MultiHeadTrainEpoch(MultiHeadEpoch):
    def __init__(self, model, loss, metrics, optimizer, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="train",
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def batch_update(self, batch):
        self.optimizer.zero_grad()
        total_loss = 0.0
        predictions = {}

        for dataset_name, mini_batch in batch.items():
            x, y = mini_batch["x"], mini_batch["y"]

            prediction = self.model.forward(x, dataset_name)
            predictions[dataset_name] = prediction

            loss = self.loss(prediction, y)
            loss /= x.size(0)

            total_loss += loss
        
        total_loss.backward()
        self.optimizer.step()

        return total_loss, predictions
    

class MultiHeadValidEpoch(MultiHeadEpoch):
    def __init__(self, model, loss, metrics, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="valid",
            device=device,
            verbose=verbose,
        )

    def batch_update(self, batch):
        with torch.no_grad():
            total_loss = 0.0
            predictions = {}

            for dataset_name, mini_batch in batch.items():
                x, y = mini_batch["x"], mini_batch["y"]

                prediction = self.model.forward(x, dataset_name)
                predictions[dataset_name] = prediction

                loss = self.loss(prediction, y)

                total_loss += loss
        return total_loss, predictions