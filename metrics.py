import logging
from typing import Union, Sequence

import torch
from ignite.metrics import EpochMetric, Metric
from ignite.exceptions import NotComputableError


class AUCPR(EpochMetric):
    def __init__(self, output_transform=lambda x: x):
        def aucpr_compute_fn(y_preds, y_targets):
            try:
                from sklearn.metrics import precision_recall_curve, auc
            except ImportError:
                raise RuntimeError("This contrib module requires sklearn to be installed.")

            y_true = y_targets.numpy()
            y_pred = y_preds.numpy()
            logging.debug(f'AUCPR {y_pred.shape}, {y_true.shape}, {sum(y_true)}')
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            return auc(recall, precision)

        super().__init__(aucpr_compute_fn, output_transform=output_transform, check_compute_fn=False)


class AP(EpochMetric):
    def __init__(self, average=None, output_transform=lambda x: x):
        def average_precision_compute_fn(y_preds, y_targets):
            try:
                from sklearn.metrics import average_precision_score
            except ImportError:
                raise RuntimeError("This contrib module requires sklearn to be installed.")

            y_true = y_targets.numpy()
            y_pred = y_preds.numpy()
            logging.debug(f'AP {y_pred.shape}, {y_true.shape}, {sum(y_true)}')
            return average_precision_score(y_true, y_pred, average=average)

        super().__init__(average_precision_compute_fn, output_transform=output_transform, check_compute_fn=False)


class AUROC(EpochMetric):
    def __init__(self, average=None, output_transform=lambda x: x):
        def auroc_compute_fn(y_preds, y_targets):
            try:
                from sklearn.metrics import roc_auc_score
            except ImportError:
                raise RuntimeError("This contrib module requires sklearn to be installed.")

            y_true = y_targets.numpy()
            y_pred = y_preds.numpy()
            return roc_auc_score(y_true, y_pred, average)

        super().__init__(auroc_compute_fn, output_transform=output_transform, check_compute_fn=False)


class Accuracy(EpochMetric):
    def __init__(self, output_transform=lambda x: x):
        def accuracy_compute_fn(y_preds, y_targets):
            try:
                from sklearn.metrics import accuracy_score
            except ImportError:
                raise RuntimeError("This contrib module requires sklearn to be installed.")

            y_true = y_targets.numpy()
            y_pred = y_preds.numpy().argmax(-1)
            return accuracy_score(y_true, y_pred)

        super().__init__(accuracy_compute_fn, output_transform=output_transform)


class Kappa(EpochMetric):
    def __init__(self, output_transform=lambda x: x):
        def kappa_compute_fn(y_preds, y_targets):
            try:
                from sklearn.metrics import cohen_kappa_score
            except ImportError:
                raise RuntimeError("This contrib module requires sklearn to be installed.")

            y_true = y_targets.numpy()
            y_pred = y_preds.numpy().argmax(-1)

            return cohen_kappa_score(y_true, y_pred)

        super().__init__(kappa_compute_fn, output_transform=output_transform)


class MicroMeanSquaredError(Metric):
    def reset(self) -> None:
        self._sum_of_squared_errors = 0.0
        self._num_examples = 0

    def update(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output
        squared_errors = torch.pow(y_pred - y.view_as(y_pred), 2)
        if squared_errors.numel() > 0:
            self._sum_of_squared_errors += torch.sum(squared_errors.mean(-1)).item()
            self._num_examples += y.shape[0]

    def compute(self) -> Union[float, torch.Tensor]:
        if self._num_examples == 0:
            raise NotComputableError("MeanSquaredError must have at least one example before it can be computed.")
        return self._sum_of_squared_errors / self._num_examples


class MicroMeanAbsoluteError(Metric):
    def reset(self) -> None:
        self._sum_of_absolute_errors = 0.0
        self._num_examples = 0

    def update(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output
        absolute_errors = torch.abs(y_pred - y.view_as(y_pred))
        if absolute_errors.numel() > 0:
            self._sum_of_absolute_errors += torch.sum(absolute_errors.mean(-1)).item()
            self._num_examples += y.shape[0]

    def compute(self) -> Union[float, torch.Tensor]:
        if self._num_examples == 0:
            raise NotComputableError("MeanAbsoluteError must have at least one example before it can be computed.")
        return self._sum_of_absolute_errors / self._num_examples


class MacroMeanSquaredError(Metric):
    def reset(self) -> None:
        self._mean_of_squared_errors = 0.0
        self._num_examples = 0

    def update(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output
        squared_errors = torch.pow(y_pred - y.view_as(y_pred), 2)
        if squared_errors.numel() > 0:
            self._mean_of_squared_errors += torch.mean(squared_errors).item()
            self._num_examples += y.shape[0]

    def compute(self) -> Union[float, torch.Tensor]:
        if self._num_examples == 0:
            raise NotComputableError("MeanSquaredError must have at least one example before it can be computed.")
        return self._mean_of_squared_errors / self._num_examples


class MacroMeanAbsoluteError(Metric):
    def reset(self) -> None:
        self._mean_of_absolute_errors = 0.0
        self._num_examples = 0

    def update(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output
        absolute_errors = torch.abs(y_pred - y.view_as(y_pred))
        if absolute_errors.numel() > 0:
            self._mean_of_absolute_errors += torch.mean(absolute_errors).item()
            self._num_examples += y.shape[0]

    def compute(self) -> Union[float, torch.Tensor]:
        if self._num_examples == 0:
            raise NotComputableError("MeanAbsoluteError must have at least one example before it can be computed.")
        return self._mean_of_absolute_errors / self._num_examples
