import logging
import os

import numpy as np
import torch
from ignite.metrics import Average, MeanAbsoluteError, MeanSquaredError
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from metrics import AP, AUCPR, AUROC, Kappa, MicroMeanAbsoluteError, MicroMeanSquaredError, MacroMeanAbsoluteError, MacroMeanSquaredError

from abc import ABC

class Label(ABC):
    def __init__(self, task, label_column, loss_weight=1.0, pos_weight=None, device='cpu', classes=None, labels=None, threshold=None):
        self.label = label_column
        self.task = task
        self.loss_weight = loss_weight
        self.device = device
        self.classes = classes
        self.labels = labels
        self.threshold = threshold
        if pos_weight is not None:
            pos_weight = torch.tensor([pos_weight]).to(device)
        self.empty_pred = torch.zeros(1, requires_grad=True).to(device)
        self.empty_true = torch.zeros(1).to(device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.results = []

    def attach_metrics(self, engine, mode, **params):
        '''Returns early stopping metric'''
        avg_loss = Average(output_transform=lambda x: x['losses'][self.task])
        avg_loss.attach(engine, f'{self.task}_loss')
        return 0

    def preprocess(self, df):
        '''df -> series'''
        return df[self.label]

    def numericalize(self, arr, device=None):
        '''numericalize label information in arr'''
        return arr

    def predict(self, score):
        return (score > self.threshold).int()

    def get_pred_true(self, output):
        return output['y_pred'][self.task], output['y_true'][self.task]

    def output_transform(self, output):
        y_pred, y_true = self.get_pred_true(output)
        return y_pred, y_true

    def predict_transform(self, output):
        y_pred, y_true = self.output_transform(output)
        y_pred = self.predict(y_pred)
        return y_pred, y_true

    def sigmoid_output_transform(self, output):
        y_pred, y_true = self.output_transform(output)
        return torch.sigmoid(y_pred), y_true

    def threshold_output_transform(self, output):
        y_pred, y_true = self.sigmoid_output_transform(output)
        return (y_pred > self.threshold).int(), y_true

    def dict_repr(self, output):
        return {}

    def loss(self, output):
        y_pred, y_true = self.output_transform(output)
        if y_true.shape[0] == 0:
            logging.debug(f'{y_pred.shape}, {y_true.shape}')
            y_pred, y_true = self.empty_pred, self.empty_true
        loss = self.loss_weight * self.criterion(y_pred, y_true)
        return loss

    def add_result(self, output):
        pass

    def _save_results(self):
        raise NotImplementedError

    def save_results(self, path, epoch):
        os.makedirs(path, exist_ok=True)
        self._save_results(f'{path}/{self.task}-epoch{epoch}.csv')
        self.results = []

    def batch(self, batch):
        return torch.stack([sample['targets'][f'{self.task}'] for sample in batch])


class MSELabel(Label):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = nn.MSELoss()

    def get_pred_true(self, output):
        y_pred = output['y_pred']['step']
        y_true = output['y_pred']['step_y']
        return y_pred, y_true

    def preprocess(self, df):
        '''df -> series'''
        return None

    def save_results(self, path, epoch):
        pass

    def numericalize(self, *arg, device=None):
        arr = torch.tensor([0])
        if device is not None:
            arr = arr.to(device)
        return arr

    def batch(self, batch):
        return torch.cat([sample['targets'][f'{self.task}'] for sample in batch])

    def attach_metrics(self, engine, mode, **params):
        super().attach_metrics(engine, mode, **params)

        def flat_transform(output):
            step, step_y = self.output_transform(output)
            return step.flatten(0, 1), step_y.flatten(0, 1)

        avg_mae = MicroMeanAbsoluteError(output_transform=flat_transform)
        avg_mae.attach(engine, 'micro_step_mae')
        avg_mae = MicroMeanSquaredError(output_transform=flat_transform)
        avg_mae.attach(engine, 'micro_step_mse')
        avg_mae = MacroMeanAbsoluteError(output_transform=self.output_transform)
        avg_mae.attach(engine, 'macro_step_mae')
        avg_mse = MacroMeanSquaredError(output_transform=self.output_transform)
        avg_mse.attach(engine, 'macro_step_mse')

        return -1 * avg_mse  # Early stopping maximises the value


class NCELabel(Label):
    def __init__(self, *args, pat_hidden_size, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = nn.CrossEntropyLoss()
        self.pat_hidden_size = pat_hidden_size

    def get_pred_true(self, output):
        y_pred = output['y_pred']['contrastive']
        y_true = output['y_pred']['contrastive_y']
        return y_pred, y_true

    def preprocess(self, df):
        '''df -> series'''
        return None

    def save_results(self, path, epoch):
        pass

    def numericalize(self, *arg, device=None):
        arr = torch.tensor([0])
        if device is not None:
            arr = arr.to(device)
        return arr

    def batch(self, batch):
        return torch.cat([sample['targets'][f'{self.task}'] for sample in batch])

    def attach_metrics(self, engine, mode, **params):
        super().attach_metrics(engine, mode, **params)

        # correct prediction if pos_score is larger than negatives.
        def transform(output):
            scores = output['y_pred']['contrastive'].detach()
            if (scores > 0).any():
                mean = (scores[:, [0]] > scores[:, 1:]).all(1).float().mean().item()
            else:
                mean = 0.
            return mean
        avg_acc = Average(output_transform=transform)
        avg_acc.attach(engine, 'contrastive_acc')

        def transform(output):
            scores = output['y_pred']['contrastive'].detach().cpu()
            if len(scores):
                return torch.where(torch.argsort(-scores) == 0)[1].float().unsqueeze(-1)
            else:
                return 0.
        avg_rank = Average(output_transform=transform)
        avg_rank.attach(engine, 'contrastive_mean_rank')

        def prediction_transform(output):
            timesteps, pat = output['timestep_prediction'], output['patient_prediction']
            if (len(timesteps) == 0) and (len(pat) == 0):
                return torch.zeros(1, 0, self.pat_hidden_size), torch.zeros(1, 0, self.pat_hidden_size)
            return torch.cat(timesteps, 1), torch.cat(pat, 1)

        def flat_prediction_transform(output):
            timesteps, pat = prediction_transform(output)
            return timesteps.flatten(0, 1), pat.flatten(0, 1)

        avg_mae = MicroMeanAbsoluteError(output_transform=flat_prediction_transform)
        avg_mae.attach(engine, 'micro_step_mae')
        avg_mae = MicroMeanSquaredError(output_transform=flat_prediction_transform)
        avg_mae.attach(engine, 'micro_step_mse')
        avg_mae = MacroMeanAbsoluteError(output_transform=prediction_transform)
        avg_mae.attach(engine, 'macro_step_mae')
        avg_mse = MacroMeanSquaredError(output_transform=prediction_transform)
        avg_mse.attach(engine, 'macro_step_mse')

        Average(output_transform=lambda x: x['losses'][self.task])\
            .attach(engine, 'contrastive_loss')
        Average(output_transform=lambda x: x['nce_accs'].cpu(), device='cpu')\
            .attach(engine, 'contrastive_accs_line_plot')
        return avg_acc


class MultiLabel(Label):
    def preprocess(self, df):
        '''df -> series'''
        return df[self.label].apply(lambda x: np.asarray(x.split(';'), dtype=np.float64, order='C'))

    def numericalize(self, arr, device=None):
        y_true = torch.from_numpy(np.array(arr))
        if device is not None:
            y_true = y_true.to(device)
        return y_true


class Phenotyping(MultiLabel):
    def sigmoid_output_transform(self, output):
        y_pred, y_true = super().sigmoid_output_transform(output)
        return y_pred, y_true

    def dict_repr(self, output):
        y_pred, y_true = self.sigmoid_output_transform(output)
        l = np.round(torch.stack([y_pred, y_true.float()], -1).detach().cpu(), 2).tolist()
        d = {}
        for i, cl in enumerate(self.classes):
            d[cl] = [sample[i] for sample in l]
        return d

    def add_result(self, output):
        y_pred, y_true = self.get_pred_true(output)
        filename, los = output['filename'], output['los']

        self.results += zip(filename, los, torch.sigmoid(y_pred).tolist(), y_true.tolist())

    def _save_results(self, path):
        n_tasks = 25
        with open(path, 'w') as f:
            header = ["stay", "period_length"]
            header += ["pred_{}".format(x) for x in range(1, n_tasks + 1)]
            header += ["label_{}".format(x) for x in range(1, n_tasks + 1)]
            header = ",".join(header)
            f.write(header + '\n')
            for name, t, pred, y in self.results:
                line = [name]
                line += ["{:.6f}".format(t)]
                line += ["{:.6f}".format(a) for a in pred]
                line += [str(a) for a in y]
                line = ",".join(line)
                f.write(line + '\n')

    def attach_metrics(self, engine, mode, **params):
        super().attach_metrics(engine, mode, **params)
        AP('macro', self.sigmoid_output_transform).attach(engine, f'{self.task}_ap_macro')
        AP('micro', self.sigmoid_output_transform).attach(engine, f'{self.task}_ap_micro')
        AUROC('micro', self.sigmoid_output_transform).attach(engine, f'{self.task}_auroc_micro')
        AUROC('weighted', self.sigmoid_output_transform).attach(engine, f'{self.task}_auroc_weighted')
        auroc = AUROC('macro', self.sigmoid_output_transform)
        auroc.attach(engine, f'{self.task}_auroc_macro')
        return auroc


class MaskedLabel(Label):
    def preprocess(self, df):
        '''df -> series'''
        def split_mask_labels(line):
            try:
                masks_labels = np.asarray(line.split(';'), dtype=np.float64, order='C')
                return masks_labels.reshape((2, -1))
            except ValueError:
                return np.array([[],[]])
        return df[self.label].apply(lambda x: split_mask_labels(x))

    def numericalize(self, arr, device=None):
        '''returns array with masks and labels'''
        # mask until 4th hour, masks are somehome wrong...
        # benchmark code does the same: Last step is masked out!
        # arr[0] = np.roll(arr[0], -1)
        arr = torch.from_numpy(np.array(arr))
        if device is not None:
            arr = arr.to(device)
        return arr

    def get_mask(self, y_pred, y_true):
        masks = y_true[:,0]
        y_pred = y_pred[:,:masks.size(1)]
        # shorten labels and masks to be the same as the predictions
        masks = masks[:,:y_pred.size(1)]
        y_true = y_true[:,1,:y_pred.size(1)].type(torch.float)
        return y_pred, y_true, masks

    def add_result(self, output):
        y_pred, y_true = self.get_pred_true(output)
        y_pred, y_true, masks = self.get_mask(y_pred, y_true)

        filenames = output['filename']

        steps = (masks == 1).sum(1).tolist()
        names = []
        for filename, step in zip(filenames, steps):
            names += [filename] * (step)

        # mask predictions and labels
        y_pred, y_true = y_pred[masks == 1], y_true[masks == 1]
        ts = np.tile(np.arange(masks.size(1)), (masks.size(0), 1))
        ts = ts[masks.cpu().numpy() == 1]

        assert len(names) == len(y_pred)
        assert len(ts) == len(y_true)

        self.results += zip(names, ts, y_pred, y_true.tolist())

    def output_transform(self, output):
        y_pred, y_true = self.get_pred_true(output)
        # shorten predictions to be same as the labels
        y_pred, y_true, masks = self.get_mask(y_pred, y_true)

        # mask predictions and labels
        y_pred, y_true = y_pred[masks == 1], y_true[masks == 1]
        return y_pred, y_true

    def batch(self, batch):
        return pad_sequence([sample['targets'][f'{self.task}'].t() for sample in batch], batch_first=True).transpose(1, 2)


class DecompensationLabel(MaskedLabel):
    def _save_results(self, path):
        with open(path, 'w') as f:
            f.write("stay,period_length,prediction,y_true\n")
            for (name, t, x, y) in self.results:
                x = torch.sigmoid(x)
                f.write("{},{:.6f},{:.6f},{}\n".format(name, t, x, y))

    def bin_output_transform(self, output):
        y_pred, y_true = self.sigmoid_output_transform(output)
        y_pred = y_pred[:, None].repeat(1, 2)
        y_pred[:, 1] = 1 - y_pred[:, 1]
        return y_pred, y_true.long()

    def attach_metrics(self, engine, mode, **params):
        super().attach_metrics(engine, mode, **params)
        AUCPR(self.sigmoid_output_transform).attach(engine, f'{self.task}_aucpr')
        AUROC(None, self.sigmoid_output_transform).attach(engine, f'{self.task}_auroc')
        ap = AP(None, self.sigmoid_output_transform)
        ap.attach(engine, f'{self.task}_ap')
        return ap


class InHospitalMortalityLabel(MultiLabel):
    def numericalize(self, arr, device=None):
        '''returns the mask and label'''
        arr = torch.from_numpy(arr[1:]).float()
        if device is not None:
            arr = arr.to(device)
        return arr

    def dict_repr(self, output):
        y_pred, y_true = self.get_pred_true(output)
        y_pred = y_pred[:,0]
        mask = y_true[:,0]
        y_true = y_true[:,1]
        y_true[mask == 0] = -1

        l = np.round(torch.stack([y_pred, y_true.float()], -1).detach().cpu(), 2).tolist()
        return {self.task: l}

    def output_transform(self, output):
        y_pred, y_true = self.get_pred_true(output)
        # logging.debug(f'{self.label}.output_transform {y_pred} {y_true}')
        y_pred = y_pred[:,0]
        mask = y_true[:,0]
        y_true = y_true[:,1]
        y_pred, y_true = y_pred[mask == 1], y_true[mask == 1]

        # logging.debug(f'{self.label}.output_transform {y_pred} {y_true}')
        return y_pred, y_true

    def add_result(self, output):
        y_pred, y_true = self.get_pred_true(output)
        filename = output['filename']

        y_pred = y_pred[:,0].cpu()
        mask = y_true[:,0].cpu()
        y_true = y_true[:,1].cpu()
        y_pred, y_true = torch.sigmoid(y_pred[mask==1]), y_true[mask==1]
        filename = np.array(filename)[mask.numpy()==1]

        self.results += zip(filename.tolist(), y_pred.tolist(), y_true.tolist())

    def _save_results(self, path):
        with open(path, 'w') as f:
            f.write("stay,prediction,y_true\n")
            for (name, x, y) in self.results:
                f.write("{},{:.6f},{}\n".format(name, x, y))

    def bin_output_transform(self, output):
        y_pred, y_true = self.sigmoid_output_transform(output)
        y_pred = y_pred[:, None].repeat(1, 2)
        y_pred[:, 1] = 1 - y_pred[:, 1]
        return y_pred, y_true.long()

    def attach_metrics(self, engine, mode, **params):
        super().attach_metrics(engine, mode, **params)
        AUCPR(self.sigmoid_output_transform).attach(engine, f'{self.task}_aucpr')
        AUROC(None, self.sigmoid_output_transform).attach(engine, f'{self.task}_auroc')
        ap = AP(None, self.sigmoid_output_transform)
        ap.attach(engine, f'{self.task}_ap')
        return ap


class LengthOfStayClassificationLabel(MaskedLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = nn.CrossEntropyLoss()
        self.empty_pred = torch.zeros(1, 10, device=self.device)
        self.empty_true = torch.zeros(1, dtype=torch.long, device=self.device)
        # start at 1 to have no negative outlier bin
        self.bins = np.array(list(range(1, 9)) + [14]) * 24

    def predict(self, score):
        return torch.argmax(score, -1)

    def numericalize(self, arr, device=None):
        '''returns masks and classification labels'''
        # arr[0] = np.roll(arr[0], -1)
        arr[1] = np.searchsorted(self.bins, arr[1], side="left")
        arr = torch.from_numpy(arr)
        if device is not None:
            arr = arr.to(device)
        return arr

    def output_transform(self, output):
        y_pred, y_true = super().output_transform(output)
        y_pred, y_true = y_pred, y_true.long()
        return y_pred, y_true

    def softmax_output_transform(self, output):
        y_pred, y_true = self.output_transform(output)
        y_pred = F.softmax(y_pred, -1)
        return y_pred, y_true

    def argmax_output_transform(self, output):
        y_pred, y_true = self.output_transform(output)
        if y_pred.shape[0] > 0:
            y_pred = y_pred.argmax(-1)
        else:
            y_pred = torch.zeros(0)
        return y_pred, y_true

    def _save_results(self, path):
        with open(path, 'w') as f:
            f.write("stay,period_length,prediction,y_true\n")
            for (name, t, x, y) in self.results:
                x = x.argmax()
                f.write("{},{:.6f},{:.6f},{:.6f}\n".format(name, t, x, y))

    def attach_metrics(self, engine, mode, **params):
        super().attach_metrics(engine, mode, **params)
        kappa = Kappa(self.output_transform)
        kappa.attach(engine, f'{self.task}_kappa')
        return kappa


class LengthOfStayRegressionLabel(MaskedLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = nn.SmoothL1Loss()

    def _save_results(self, path):
        with open(path, 'w') as f:
            f.write("stay,period_length,prediction,y_true\n")
            for (name, t, x, y) in self.results:
                f.write("{},{:.6f},{:.6f},{:.6f}\n".format(name, t, x, y))

    def attach_metrics(self, engine, mode, **params):
        super().attach_metrics(engine, mode, **params)
        abs = MeanAbsoluteError(self.output_transform)
        abs.attach(engine, f'{self.task}_mad')
        MeanSquaredError(self.output_transform).attach(engine, f'{self.task}_mse')
        return 0
        # return -1 * abs


def get_labels(device,
               tasks=['ihm', 'decomp', 'los_reg', 'los_cl', 'phen'],
               loss_weight_decomp=2.,
               loss_weight_ihm=.2,
               loss_weight_los_cl=1.,
               loss_weight_los_reg=.1,
               loss_weight_phen=1.,
               loss_weight_contr=1.,
               pat_hidden_size=128,
               **kwargs):
    all_tasks = {
        'decomp': DecompensationLabel(task='decompensation',
                                      label_column='decompensation task (masks;labels)',
                                      loss_weight=loss_weight_decomp,
                                      pos_weight=50.,
                                      labels=np.arange(2),
                                      classes=['lives', 'dies'],
                                      threshold=0.95,
                                      device=device),
        'ihm': InHospitalMortalityLabel(task='in_hospital_mortality',
                                        label_column='in-hospital mortality task (pos;mask;label)',
                                        loss_weight=loss_weight_ihm,
                                        pos_weight=10.,
                                        labels=np.arange(2),
                                        classes=['lives', 'dies'],
                                        threshold=0.8,
                                        device=device),
        'los_cl': LengthOfStayClassificationLabel(task='length_of_stay_classification',
                                                  label_column='length of stay task (masks;labels)',
                                                  loss_weight=loss_weight_los_cl,
                                                  labels=np.arange(10),
                                                  classes=['(0, 1)', '(1, 2)', '(2, 3)', '(3, 4)', '(4, 5)', '(5, 6)', '(6, 7)', '(7, 8)', '(8, 14)', '14 +'],
                                                  device=device),
        'los_reg': LengthOfStayRegressionLabel(task='length_of_stay_regression',
                                               label_column='length of stay task (masks;labels)',
                                               loss_weight=loss_weight_los_reg,
                                               device=device),
        'phen': Phenotyping(task='phenotyping',
                            label_column='phenotyping task (labels)',
                            loss_weight=loss_weight_phen,
                            labels=np.arange(25),
                            classes=['Acute and unspecified renal failure',
                                     'Acute cerebrovascular disease',
                                     'Acute myocardial infarction',
                                     'Cardiac dysrhythmias',
                                     'Chronic kidney disease',
                                     'Chronic obstructive pulmonary disease',
                                     'Complications of surgical/medical care',
                                     'Conduction disorders',
                                     'Congestive heart failure; nonhypertensive',
                                     'Coronary atherosclerosis and related',
                                     'Diabetes mellitus with complications',
                                     'Diabetes mellitus without complication',
                                     'Disorders of lipid metabolism',
                                     'Essential hypertension',
                                     'Fluid and electrolyte disorders',
                                     'Gastrointestinal hemorrhage',
                                     'Hypertension with complications',
                                     'Other liver diseases',
                                     'Other lower respiratory disease',
                                     'Other upper respiratory disease',
                                     'Pleurisy; pneumothorax; pulmonary collapse',
                                     'Pneumonia',
                                     'Respiratory failure; insufficiency; arrest',
                                     'Septicemia (except in labor)',
                                     'Shock'],
                            device=device),
        'contr': NCELabel(task='contrastive',
                          label_column='',
                          loss_weight=loss_weight_contr,
                          labels=np.arange(2),
                          classes=['wrong_pred', 'correct_pred'],
                          threshold=0.95,
                          device=device,
                          pat_hidden_size=pat_hidden_size),
        'pred': MSELabel(task='pred',
                         label_column='',
                         loss_weight=1,
                         device=device)
    }

    return {label.task: label for key, label in all_tasks.items() 
            if key in tasks}


if __name__ == '__main__':
    from metrics import *
    from ignite.metrics import *

    def test_label(label, y_true, y_pred):
        print(label.task)
        y_true = label.numericalize(y_true)
        print('y_true', y_true)
        print('y_pred', torch.sigmoid(y_pred))
        loss = label.loss(y_pred={label.task: y_pred}, y_true={label.task: y_true})
        print('loss', loss.item())
        return {'y_true': {label.task: y_true}, 'y_pred': {label.task: y_pred}}

    def test_metric(metric, output_function, output):
        metric.update(output_function(output))
        print(metric.__class__.__name__, metric.compute())

    # def inv_sigmoid(sigmoid):
    #     return -np.log((1./(sigmoid+1e-5)) - 1.)

    task_ihm = InHospitalMortalityLabel(task='in_hospital_mortality', loss_weight=0.2, label_column='in-hospital mortality task (pos;mask;label)', pos_weight=5.)
    outputs = []
    y_true = np.array([47.,1.,0.])
    y_pred = torch.tensor([[100.]])
    output = test_label(task_ihm, y_true, y_pred)
    test_metric(AP(), task_ihm.sigmoid_output_transform, output)

    task_decomp = MaskedLabel(task='decompensation', loss_weight=1.0, label_column='decompensation task (masks;labels)', pos_weight=10.)
    y_true = (np.array([0., 0., 1., 1., 1., 1., 1.]), np.array([1., 1., 0., 0., 0., 1., 1.]))
    y_pred = torch.tensor([[0., 0., -100., -100., -100., 100., 100]])[:,:,None]
    output = test_label(task_decomp, y_true, y_pred)
    test_metric(AP(), task_decomp.sigmoid_output_transform, output)
    test_metric(AUROC(), task_decomp.sigmoid_output_transform, output)

    y_pred = torch.tensor([[0., 0., -100., -100., -100., 100., -100]])[:,:,None]
    output = test_label(task_decomp, y_true, y_pred)
    test_metric(AP(), task_decomp.sigmoid_output_transform, output)
    test_metric(AUROC(), task_decomp.sigmoid_output_transform, output)

    # phenotyping_data = {'y_true': {'phenotyping': np.array([0., 0., 0., 0., 0.])}}
    # phenotyping_masked_data = {'y_true': {'phenotyping': np.array([0., 0., 0., 0., 0.])}}
    # length_of_stay_classification_data = {'y_true': {'length_of_stay_classification': (np.array([0., 0., 1., 1.]), np.array([9., 9., 8., 7.]))}}

    # length_of_stay_regression_data = {'y_true': {'length_of_stay_regression': (np.array([0., 0., 1., 1.]), np.array([150., 120., 100., 91.]))}}
