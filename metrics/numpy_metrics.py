import torch
import numpy as np
from sklearn.metrics import confusion_matrix

from utils.distributed_utils import all_reduce_mean, get_current_device

np.seterr(divide='ignore', invalid='ignore')


class RunningMetrics(object):

    def __init__(self, num_classes: int, ignore_index: int) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        self.conf_matrix = np.zeros((num_classes, num_classes))


    def update(self, logits: torch.tensor, labels: torch.tensor) -> None:
        _, preds = torch.max(logits, dim=1)

        preds = preds.reshape(-1).detach().cpu().numpy()
        labels = labels.reshape(-1).detach().cpu().numpy()

        # Filtering unwanted class
        if self.ignore_index is not None:
            mask = (labels != self.ignore_index)

            preds = preds[mask]
            labels = labels[mask]

        self.conf_matrix += confusion_matrix(
            y_pred=preds, y_true=labels, labels=[i for i in range(self.num_classes)]
        )


    def get_scores(self) -> dict[str, float]:
        metrics = [
            'pixel_acc',
            'mean_class_acc',
            'mean_precision',
            'mean_recall',
            'mean_f1',
            'mean_iou'
        ]
        device = get_current_device()
        metrics_tensor = torch.zeros(len(metrics)).to(device)

        diag = np.diagonal(self.conf_matrix)

        row_sum = self.conf_matrix.sum(axis=1)
        col_sum = self.conf_matrix.sum(axis=0)

        # Pixel accuracy
        metrics_tensor[0] = np.nan_to_num(diag.sum() / self.conf_matrix.sum()).item()

        # Mean class accuracy
        class_acc = np.nan_to_num(diag / self.conf_matrix.sum(axis=1))
        metrics_tensor[1] = np.mean(class_acc).item()

        # Mean precision
        precision = np.nan_to_num(diag / col_sum)
        metrics_tensor[2] = np.mean(precision).item()

        # Mean recall
        recall = np.nan_to_num(diag / row_sum)
        metrics_tensor[3] = np.mean(recall).item()

        # Mean F1
        f1 = np.nan_to_num((2*precision*recall) / (precision + recall))
        metrics_tensor[4] = np.mean(f1).item()

        # Mean IoU
        iou = np.nan_to_num(diag / (row_sum + col_sum - diag))
        metrics_tensor[5] = np.mean(iou).item()

        all_reduce_mean(metrics_tensor)

        return {metric: metrics_tensor[i].item() for i, metric in enumerate(metrics)}


    def reset(self) -> None:
        self.conf_matrix = np.zeros((self.num_classes, self.num_classes))