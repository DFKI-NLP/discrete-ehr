import torch
from torch import nn
from torch.nn import functional as F
from utils import load_class

from models import modules
from dataloader.data import TabularFeature, JointTabularFeature


class MultitaskFinetune(nn.Module):
    def __init__(self, joint_vocab, tables, timestep_agg_modelcls='models.TimestepEncoder', patient_modelcls='models.PatientRNNEncoder', **kwargs):
        super().__init__()

        tabular_tables = [table for table in tables if isinstance(table, (TabularFeature, JointTabularFeature))]
        self.timestep_encoder = load_class(timestep_agg_modelcls)(joint_vocab, tables=tabular_tables, **kwargs)
        self.patient_encoder = load_class(patient_modelcls)(self.timestep_encoder.out_size, tables=tabular_tables, **kwargs)
        self.predictor = modules.BenchmarkPredictor(self.patient_encoder.out_size, **kwargs)

    def forward(self, dem, *tables):
        '''
        Input:
            chartevents: Tensor for chart table with shape: N, L, C
            labevents: Tensor for chart table with shape: N, L, C
            outputevents: Tensor for chart table with shape: N, L, C
            dem: Tensor for demographics with shape: N, C
        Output:
            Tuple of (Dictionary of predictions, Embeddings)
        '''
        outputs = {}
        timesteps, output = self.timestep_encoder(*tables)  # N, L, C
        outputs.update(output)

        patient_timesteps, output = self.patient_encoder(dem, *timesteps)  # N, L, C
        outputs.update(output)
        preds, _ = self.predictor(patient_timesteps)
        outputs.update({'patient': patient_timesteps.detach(),
                        'timesteps': [timestep.detach() for timestep in timesteps]})
        return preds, outputs


class MultitaskMultiVariateFinetune(nn.Module):
    def __init__(self, joint_vocab, tables, patient_modelcls='models.PatientRNNEncoder', **kwargs):
        super().__init__()

        tabular_tables = [table for table in tables if isinstance(table, (TabularFeature, JointTabularFeature))]
        self.patient_encoder = load_class(patient_modelcls)(2 * len(joint_vocab), tables=tabular_tables, **kwargs)
        self.predictor = modules.BenchmarkPredictor(self.patient_encoder.out_size, **kwargs)

    def forward(self, dem, *tables):
        '''
        Input:
            chartevents: Tensor for chart table with shape: N, L, C
            labevents: Tensor for chart table with shape: N, L, C
            outputevents: Tensor for chart table with shape: N, L, C
            dem: Tensor for demographics with shape: N, C
        Output:
            Tuple of (Dictionary of predictions, Embeddings)
        '''
        outputs = {}

        input = torch.cat(tables, -1)
        patient_timesteps, output = self.patient_encoder(input, dem)  # N, L, C
        outputs.update(output)
        preds, _ = self.predictor(patient_timesteps)
        outputs.update({'patient': patient_timesteps.detach()})
        return preds, outputs


class ContrastiveModel(nn.Module):
    def __init__(self, joint_vocab, tables, timestep_agg_modelcls='models.TimestepEncoder', patient_modelcls='models.PatientRNNEncoder', prediction_steps=1, temperature=0.07, **kwargs):
        super().__init__()

        tabular_tables = [table for table in tables if isinstance(table, TabularFeature) or isinstance(table, JointTabularFeature)]
        self.timestep_encoder = load_class(timestep_agg_modelcls)(joint_vocab, tables=tabular_tables, **kwargs)
        self.patient_encoder = load_class(patient_modelcls)(self.timestep_encoder.out_size, tables=tabular_tables, **kwargs)

        self.step_predictors = nn.ModuleList([nn.Linear(self.patient_encoder.out_size, self.timestep_encoder.out_size) for _ in range(1, prediction_steps+1)])
        self.prediction_steps = range(1, prediction_steps+1)
        self.temperature = temperature

    def forward(self, dem, *tables):
        '''
        Input:
            chartevents: Tensor for chart table with shape: N, L, C
            labevents: Tensor for chart table with shape: N, L, C
            outputevents: Tensor for chart table with shape: N, L, C
            dem: Tensor for demographics with shape: N, C
        Output:
            Tuple of (Dictionary of predictions, Embeddings)
            Dictionary of predictions:
                Logits for similarity scores with shape (N L)xB where positive samples are in scores[:, 0] and rest are negative sample.
        '''

        timesteps, _ = self.timestep_encoder(*tables)  # N, L, C

        patient_timesteps, _ = self.patient_encoder(dem, *timesteps)  # N, L, C
        N = patient_timesteps.shape[0]

        logits = []
        accs = []
        for (prediction_step, step_predictor) in zip(self.prediction_steps, self.step_predictors):
            _patient_timesteps = patient_timesteps[:, :-prediction_step]
            _timesteps = timesteps[:, prediction_step:]
            
            prediction = step_predictor(_patient_timesteps)
            if prediction.shape[1] == 0:
                # l_pos = torch.zeros(prediction.shape[:2], device=prediction.device)
                l_neg = torch.zeros(prediction.shape[0], prediction.shape[0], 0, device=prediction.device)
            else:
                # l_pos = torch.einsum('nlc,nlc->nl', [_timesteps, prediction])
                l_neg = torch.einsum('nlc,mlc->nml', [_timesteps, prediction])  # N, N, L
           
            acc = []
            for i in range(N):
                n = l_neg.detach()
                acc.append(((n[i,i] > n[i, :i]).all(0) & (n[i,i] > n[i, i+1:]).all(0)).float().cpu())
            accs.append(torch.mean(torch.cat(acc)))

            logit = l_neg / self.temperature
            logits.append(logit)

        accs = torch.stack(accs)
        logits = torch.cat(logits, -1)
        y_trues = torch.arange(N, device=logits.device)[:, None].repeat(1, logits.shape[-1])
        return {'contrastive': logits,
                'contrastive_y': y_trues}, {'nce_accs': accs,
                                            'patient': patient_timesteps,
                                            'timesteps': timesteps}


class PredictionRNNModel(nn.Module):
    def __init__(self, joint_vocab, tables, timestep_agg_modelcls='models.TimestepEncoder', patient_modelcls='models.PatientRNNEncoder', pat_hidden_size=1000, prediction_steps=[1], predictor=False, predictor_mlp=False, **kwargs):
        super().__init__()

        tabular_tables = [table for table in tables if isinstance(table, TabularFeature) or isinstance(table, JointTabularFeature)]
        self.timestep_encoder = load_class(timestep_agg_modelcls)(joint_vocab, tables=tabular_tables, **kwargs)

        if predictor:
            self.patient_encoder = load_class(patient_modelcls)(self.timestep_encoder.out_size, pat_hidden_size=pat_hidden_size, tables=tabular_tables, **kwargs)
            if predictor_mlp:
                self.step_predictors = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(self.patient_encoder.out_size, 2 * 128),
                        nn.ReLU(),
                        nn.Linear(2 * 128, self.patient_encoder.out_size)
                        )
                    for _ in prediction_steps])
            else:
                self.step_predictors = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(self.patient_encoder.out_size, self.patient_encoder.out_size),
                        )
                    for _ in prediction_steps])
        else:
            pat_hidden_size = self.timestep_encoder.out_size
            self.patient_encoder = load_class(patient_modelcls)(self.timestep_encoder.out_size, pat_hidden_size=pat_hidden_size, tables=tabular_tables, **kwargs)

        if len(prediction_steps) == 0:
            raise ValueError
        if len(prediction_steps) > 1:
            raise NotImplementedError
        self.prediction_steps = prediction_steps
        self.predictor = predictor

    def forward(self, dem, *tables):
        '''
        Input:
            chartevents: Tensor for chart table with shape: N, L, C
            labevents: Tensor for chart table with shape: N, L, C
            outputevents: Tensor for chart table with shape: N, L, C
            dem: Tensor for demographics with shape: N, C
        Output:
            Tuple of (Dictionary of predictions, Embeddings)
            Dictionary of predictions:
                Logits for similarity scores with shape (N L)xB where positive samples are in scores[:, 0] and rest are negative sample.
        '''

        timesteps, _ = self.timestep_encoder(*tables)  # N, L, C

        patient_timesteps, _ = self.patient_encoder(dem, *timesteps)  # N, L, C

        _patient_timesteps = patient_timesteps[:, :-self.prediction_steps[0]]

        if self.predictor:
            predicted_ts = self.step_predictors[0](_patient_timesteps)
        else:
            predicted_ts = _patient_timesteps

        extra = {'patient': patient_timesteps,
                 'timesteps': timesteps}
        y_pred = {'step': predicted_ts,
                  'step_y': patient_timesteps[self.prediction_steps[0]:]}

        return y_pred, extra


class PredictionModel(nn.Module):
    def __init__(self, joint_vocab, tables, timestep_agg_modelcls='models.TimestepEncoder', patient_modelcls='models.PatientRNNEncoder', pat_hidden_size=1000, prediction_steps=[1], predictor=False, predictor_mlp=False, **kwargs):
        super().__init__()

        tabular_tables = [table for table in tables if isinstance(table, TabularFeature) or isinstance(table, JointTabularFeature)]
        self.timestep_encoder = load_class(timestep_agg_modelcls)(joint_vocab, tables=tabular_tables, **kwargs)

        if predictor:
            self.patient_encoder = load_class(patient_modelcls)(self.timestep_encoder.out_size, pat_hidden_size=pat_hidden_size, tables=tabular_tables, **kwargs)
            if predictor_mlp:
                self.step_predictors = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(self.patient_encoder.out_size, 2 * 128),
                        nn.ReLU(),
                        nn.Linear(2 * 128, self.timestep_encoder.out_size)
                        )
                    for _ in prediction_steps])
            else:
                self.step_predictors = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(self.patient_encoder.out_size, self.timestep_encoder.out_size),
                        )
                    for _ in prediction_steps])
        else:
            pat_hidden_size = self.timestep_encoder.out_size
            self.patient_encoder = load_class(patient_modelcls)(self.timestep_encoder.out_size, pat_hidden_size=pat_hidden_size, tables=tabular_tables, **kwargs)

        if len(prediction_steps) == 0:
            raise ValueError
        if len(prediction_steps) > 1:
            raise NotImplementedError
        self.prediction_steps = prediction_steps
        self.predictor = predictor

    def forward(self, dem, *tables):
        '''
        Input:
            chartevents: Tensor for chart table with shape: N, L, C
            labevents: Tensor for chart table with shape: N, L, C
            outputevents: Tensor for chart table with shape: N, L, C
            dem: Tensor for demographics with shape: N, C
        Output:
            Tuple of (Dictionary of predictions, Embeddings)
            Dictionary of predictions:
                Logits for similarity scores with shape (N L)xB where positive samples are in scores[:, 0] and rest are negative sample.
        '''

        timesteps, _ = self.timestep_encoder(*tables)  # N, L, C

        patient_timesteps, _ = self.patient_encoder(dem, *timesteps)  # N, L, C
        ts = torch.cat(timesteps, -1)

        _patient_timesteps = patient_timesteps[:, :-self.prediction_steps[0]]
        true_ts = ts[:, self.prediction_steps[0]:]

        if self.predictor:
            predicted_ts = self.step_predictors[0](_patient_timesteps)
        else:
            predicted_ts = _patient_timesteps

        extra = {'patient': patient_timesteps,
                 'timesteps': timesteps}
        y_pred = {'step': predicted_ts,
                  'step_y': true_ts}

        return y_pred, extra


class MomentumContrastiveModel(nn.Module):
    def __init__(self, joint_vocab, tables, predictor_mlp=True, timestep_agg_modelcls='models.TimestepEncoder', patient_modelcls='models.PatientRNNEncoder', prediction_steps=[1], K=10000, momentum=0.999, temperature=0.07, cossim=False, auxiliary=False, **kwargs):
        super().__init__()

        tabular_tables = [table for table in tables if isinstance(table, TabularFeature) or isinstance(table, JointTabularFeature)]
        self.timestep_encoder = load_class(timestep_agg_modelcls)(joint_vocab, tables=tabular_tables, **kwargs)
        self.patient_encoder = load_class(patient_modelcls)(self.timestep_encoder.out_size, tables=tabular_tables, **kwargs)

        if predictor_mlp:
            self.step_predictors_pat = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.patient_encoder.out_size, 2 * 128),
                    nn.ReLU(),
                    nn.Linear(2 * 128, 128)
                    )
                for _ in prediction_steps])

            self.step_predictors_ts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.timestep_encoder.out_size, 2 * 128),
                    nn.ReLU(),
                    nn.Linear(2 * 128, 128)
                    )
                for _ in prediction_steps])
        else:
            self.step_predictors_pat = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.patient_encoder.out_size, 128),
                    )
                for _ in prediction_steps])

            self.step_predictors_ts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.timestep_encoder.out_size, 128),
                    )
                for _ in prediction_steps])

        self.prediction_steps = prediction_steps
        self.temperature = temperature

        self.momentum_timestep_encoder = load_class(timestep_agg_modelcls)(joint_vocab, tables=tabular_tables, **kwargs)
        for param_q, param_k in zip(self.timestep_encoder.parameters(), self.momentum_timestep_encoder.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.register_buffer("queue", torch.randn(K, self.timestep_encoder.out_size))
        self.queue = nn.functional.normalize(self.queue, dim=1)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        if auxiliary:
            self.predictor = modules.BenchmarkPredictor(self.patient_encoder.out_size, **kwargs)

        self.auxiliary = auxiliary
        self.K = K
        self.m = momentum
        self.temperature = temperature
        self.cossim = cossim

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.timestep_encoder.parameters(), self.momentum_timestep_encoder.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        ptr = int(self.queue_ptr)
        # do not handle overflow for simplicity
        batch_size = min(keys.shape[0], self.K - ptr)

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size] = keys[:batch_size]
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, dem, *tables):
        outputs = {}
        
        timesteps, _ = self.timestep_encoder(*tables)  # N, L, C
        patient_timesteps, _ = self.patient_encoder(dem, *timesteps)  # N, L, C

        # compute key tables
        with torch.no_grad():  # no gradient to keys
            m_timesteps, _ = self.momentum_timestep_encoder(*tables)
            m_timesteps = torch.cat(m_timesteps, -1)

        logits = []
        accs = []
        ts_predictions = []
        pat_predictions = []
        for (prediction_step, pat_predictor, ts_predictor) in zip(self.prediction_steps, self.step_predictors_pat, self.step_predictors_ts):
            _patient_timesteps = patient_timesteps[:, :-prediction_step]
            _timesteps = m_timesteps[:, prediction_step:]

            pat_prediction = pat_predictor(_patient_timesteps)
            ts_prediction = ts_predictor(_timesteps)

            if self.cossim:
                pat_prediction = F.normalize(pat_prediction, dim=-1)
                ts_prediction = F.normalize(ts_prediction, dim=-1)

            if pat_prediction.shape[1] == 0:
                l_pos = torch.zeros(pat_prediction.shape[0], 0, 1, device=pat_prediction.device)
                l_neg = torch.zeros(pat_prediction.shape[0], 0, self.K, device=pat_prediction.device)
                # Choosing to underestimating performance, happens seldom
                accs.append(torch.tensor([0.], device=pat_prediction.device))
            else:
                l_pos = torch.einsum('nlc,nlc->nl', [ts_prediction, pat_prediction]).unsqueeze(-1)  # N x L x 1
                ts_predictions.append(ts_prediction.detach())
                pat_predictions.append(pat_prediction.detach())

                mk_predictions = ts_predictor(self.queue.clone().detach())  # K x C
                if self.cossim:
                    mk_predictions = F.normalize(mk_predictions, dim=-1)

                # N x L x K
                l_neg = torch.einsum('nlc,kc->nlk', [pat_prediction, mk_predictions])

                acc = (l_pos.detach() > l_neg.detach()).all(-1).float().mean()
                accs.append(acc[None])

            logit = torch.cat([l_pos, l_neg], dim=-1).flatten(0, 1) / self.temperature
            logits.append(logit)

        accs = torch.stack(accs, 1)
        logits = torch.cat(logits, 0)

        # dequeue and enqueue
        self._dequeue_and_enqueue(m_timesteps.flatten(0, 1))
        outputs = {'nce_accs': accs,
                   'patient': patient_timesteps,
                   'timesteps': timesteps,
                   'timesteps_momentum': m_timesteps,
                   'timestep_prediction': ts_predictions,
                   'patient_prediction': pat_predictions}
        y_preds = {'contrastive': logits,
                   'contrastive_y': torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)}

        if self.auxiliary:
            preds, patient_timesteps = self.predictor(patient_timesteps)
            y_preds.update(preds)
            outputs.update({'patient': patient_timesteps})
        return y_preds, outputs
