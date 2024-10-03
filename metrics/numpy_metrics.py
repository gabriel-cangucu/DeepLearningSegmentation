import torch
import numpy as np
from sklearn.metrics import confusion_matrix

np.seterr(divide='ignore', invalid='ignore')


class RunningMetrics(object):

    def __init__(self, num_classes: int, ignore_index: int, device: torch.device) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.device = device

        self.conf_matrix = torch.zeros(num_classes, num_classes).to(device)


    def update(self, logits: torch.tensor, labels: torch.tensor) -> None:
        _, preds = torch.max(logits, dim=1)

        preds = preds.reshape(-1).detach().cpu().numpy()
        labels = labels.reshape(-1).detach().cpu().numpy()

        # Filtering unwanted class
        if self.ignore_index is not None:
            mask = (labels != self.ignore_index)

            preds = preds[mask]
            labels = labels[mask]

        self.conf_matrix += torch.tensor(
            confusion_matrix(y_pred=preds, y_true=labels, labels=[i for i in range(self.num_classes)])
        ).to(self.device)


    def get_scores(self) -> dict[str, float]:
        torch.distributed.all_reduce(self.conf_matrix, op=torch.distributed.ReduceOp.AVG)

        conf_matrix = self.conf_matrix.detach().cpu()
        diag = np.diagonal(conf_matrix)

        row_sum = conf_matrix.sum(axis=1)
        col_sum = conf_matrix.sum(axis=0)

        pixel_acc = np.nan_to_num(diag.sum() / conf_matrix.sum()).item()
        class_acc = np.nan_to_num(diag / conf_matrix.sum(axis=1))
        mean_class_acc = np.mean(class_acc).item()

        precision = np.nan_to_num(diag / col_sum)
        mean_precision = np.mean(precision).item()
        recall = np.nan_to_num(diag / row_sum)
        mean_recall = np.mean(recall).item()

        f1 = np.nan_to_num((2*precision*recall) / (precision + recall))
        mean_f1 = np.mean(f1).item()

        iou = np.nan_to_num(diag / (row_sum + col_sum - diag))
        mean_iou = np.mean(iou).item()

        return {
            'pixel_acc': pixel_acc,
            'mean_class_acc': mean_class_acc,
            'mean_precision': mean_precision,
            'mean_recall': mean_recall,
            'mean_f1': mean_f1,
            'mean_iou': mean_iou
        }


    def reset(self) -> None:
        self.conf_matrix = np.zeros((self.num_classes, self.num_classes))