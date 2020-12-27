from dataloader.data import TextTabularFeature
import torch
import torch.nn

from models.modules import *
from utils import load_class
from dataloader.utils import get_vocab


class AdditiveEventEncoder(nn.Module):
    def __init__(self, vocab, freeze_emb, emb_dim, include_time, n_bins, **kwargs):
        super().__init__()

        self.vocab = vocab
        if vocab.vectors is not None:
            self.encoder = nn.Embedding.from_pretrained(vocab.vectors, freeze=freeze_emb, padding_idx=0)
        else:
            self.encoder = nn.Embedding(len(vocab), emb_dim, padding_idx=0)
            if freeze_emb:
                self.encoder.weight.requires_grad = False

        self.bins = nn.Embedding(n_bins + 1, self.encoder.embedding_dim, scale_grad_by_freq=True)

        self.input_size = self.encoder.weight.shape[1]
        if include_time:
            self.input_size += 2

        self.include_time = include_time

    def forward(self, input):
        if input.shape[-1] == 100:
            emb = self.encoder(input)
        else:
            emb = self.encoder(input[:, :, 0])

        bin_ixs = input[:, :, 1].long()
        bins = self.bins(bin_ixs)
        emb = emb + bins

        arr = [emb]
        if self.include_time:
            time_input = torch.arange(len(input)).float()[:, None, None].expand(input.shape[:2] + (1,)).to(next(self.parameters()).device)
            arr.extend([torch.log(time_input + 1), torch.exp(time_input/1000) - 1])
        emb = torch.cat(arr, -1)  # N, L, C
        return emb


class AdditiveTableEventEncoder(nn.Module):
    def __init__(self, vocab, freeze_emb, emb_dim, include_time, value_vocab, **kwargs):
        super().__init__()

        self.vocab = vocab
        if vocab.vectors is not None:
            self.encoder = nn.Embedding.from_pretrained(vocab.vectors, freeze=freeze_emb, padding_idx=0)
        else:
            self.encoder = nn.Embedding(len(vocab), emb_dim, padding_idx=0)
            if freeze_emb:
                self.encoder.weight.requires_grad = False

        self.value_vocab = get_vocab('embeddings/sentences.mimic3.train.kmeans.10bins.clo.merged.train.txt.Fasttext.vec',
                                     vocab_file='embeddings/sentences.mimic3.train.kmeans.10bins.clo.merged.train.counts')
        self.values = nn.Embedding.from_pretrained(self.value_vocab.vectors, freeze=freeze_emb, padding_idx=0, scale_grad_by_freq=True)
        self.value_linear = nn.Linear(vocab.vectors.size(-1), vocab.vectors.size(-1))
        self.label_linear = nn.Linear(vocab.vectors.size(-1), vocab.vectors.size(-1))

        self.input_size = self.encoder.weight.shape[1]
        if include_time:
            self.input_size += 2

        self.include_time = include_time

    def forward(self, input):
        if input.shape[-1] == 100:
            emb = self.encoder(input)
        else:
            emb = self.encoder(input[:, :, 0])

        value_ixs = input[:, :, 1].long()
        values = self.values(value_ixs)

        emb = F.relu(self.label_linear(emb))
        values = F.relu(self.value_linear(values))

        emb = emb + values

        arr = [emb]
        if self.include_time:
            time_input = torch.arange(len(input)).float()[:, None, None].expand(input.shape[:2] + (1,)).to(input.device)
            arr.extend([torch.log(time_input + 1), torch.exp(time_input/1000) - 1])
        emb = torch.cat(arr, -1)  # N, L, C
        return emb


class FixedEventEncoder(nn.Module):
    def __init__(self, vocab, freeze_emb, emb_dim, include_time, append_bin, n_bins, **kwargs):
        super().__init__()

        self.vocab = vocab
        if vocab.vectors is not None:
            vocab.vectors[1:] = torch.randn(vocab.vectors.shape[-1])
            self.encoder = nn.Embedding.from_pretrained(vocab.vectors, freeze=freeze_emb, padding_idx=0)
        else:
            self.encoder = nn.Embedding(len(vocab), emb_dim, padding_idx=0)
            if freeze_emb:
                self.encoder.weight.requires_grad = False

        if append_bin:
            bins = torch.eye(n_bins)
            bins = torch.cat([torch.zeros(1, n_bins), bins], 0)
            self.bins = nn.Parameter(bins, requires_grad=False)

        self.input_size = self.encoder.weight.shape[1]
        if append_bin:
            self.input_size += n_bins
        if include_time:
            self.input_size += 2

        self.include_time = include_time
        self.append_bin = append_bin

    def forward(self, input):
        if input.shape[-1] == 100:
            emb = self.encoder(input)
        else:
            emb = self.encoder(input[:, :, 0])
        arr = [emb]
        if self.include_time:
            time_input = torch.arange(len(input)).float()[:, None, None].expand(input.shape[:2] + (1,)).to(next(self.parameters()).device)
            arr.extend([torch.log(time_input + 1), torch.exp(time_input/1000) - 1])
        if self.append_bin:
            bins = self.bins[input[:, :, 1].long()]
            arr.append(bins)
        emb = torch.cat(arr, -1)  # N, L, C
        return emb


class EventEncoder(nn.Module):
    def __init__(self, vocab, freeze_emb, emb_dim, include_time, append_bin, n_bins, **kwargs):
        super().__init__()

        self.vocab = vocab
        if vocab.vectors is not None:
            self.encoder = nn.Embedding.from_pretrained(vocab.vectors, freeze=freeze_emb, padding_idx=0)
        else:
            self.encoder = nn.Embedding(len(vocab), emb_dim, padding_idx=0)
            if freeze_emb:
                self.encoder.weight.requires_grad = False

        if append_bin:
            bins = torch.eye(n_bins)
            bins = torch.cat([torch.zeros(1, n_bins), bins], 0)
            self.bins = nn.Parameter(bins, requires_grad=False)

        self.input_size = self.encoder.weight.shape[1]
        if append_bin:
            self.input_size += n_bins
        if include_time:
            self.input_size += 2

        self.include_time = include_time
        self.append_bin = append_bin

    def forward(self, input):
        if input.shape[-1] == 100:
            emb = self.encoder(input)
        else:
            emb = self.encoder(input[:, :, 0])
        arr = [emb]
        if self.include_time:
            time_input = torch.arange(len(input)).float()[:, None, None].expand(input.shape[:2] + (1,)).to(next(self.parameters()).device)
            arr.extend([torch.log(time_input + 1), torch.exp(time_input/1000) - 1])
        if self.append_bin:
            bins = self.bins[input[:, :, 1].long()]
            arr.append(bins)
        emb = torch.cat(arr, -1)  # N, L, C
        return emb


class SeperateTimestepEncoder(nn.Module):
    def __init__(self, joint_vocab, tables, hidden_size=50,
                 event_modelcls='models.EventEncoder',
                 timestep_modelcls='models.LinearMaxMeanSumPool',
                 freeze_encoders=False,
                 value_vocab=None,
                 **kwargs):
        super().__init__()

        self.event_encoder = load_class(event_modelcls)(joint_vocab, value_vocab=value_vocab, **kwargs)

        models = []
        self.out_size = 0
        for table in tables:
            if isinstance(table, TextTabularFeature):
                model = load_class(timestep_modelcls)(table.n_dims, hidden_size=hidden_size, **kwargs)
            else:
                model = load_class(timestep_modelcls)(self.event_encoder.input_size, hidden_size=hidden_size, **kwargs)
            models.append(model)
            self.out_size += model.out_size
        self.models = nn.ModuleList(models)

        if freeze_encoders:
            for param in self.parameters():
                param.requires_grad = False

        self.tables = tables

    def forward(self, *inputs: Tensor):
        '''
        Input:
            inputs: Tensors for table with shape: N, L, C
        Output:
            Tensor of timestep embeddings with shape: N, L, C
        '''
        out = []
        measure_inds = {}
        for i, (table, input, model) in enumerate(zip(self.tables, inputs, self.models)):
            batch_size = input.size(0)
            input = input.reshape((-1, ) + input.shape[2:])  # NxL1, L2, C
            if not isinstance(table, TextTabularFeature):
                input = self.event_encoder(input)
            input, measure_ind = model(input)  # NxL, C
            measure_inds[table.table] = measure_ind

            input = input.reshape((batch_size, -1) + input.shape[1:])  # N, L, C
            out.append(input)

        return out, {'measure_inds': measure_inds}


class TimestepEncoder(nn.Module):
    def __init__(self, joint_vocab, tables, hidden_size=50,
                 event_modelcls='models.EventEncoder',
                 timestep_modelcls='models.LinearMaxMeanSumPool', freeze_encoders=False,
                 value_vocab=None,
                 **kwargs):
        super().__init__()

        self.event_encoder = load_class(event_modelcls)(joint_vocab, value_vocab=value_vocab, **kwargs)
        self.model = load_class(timestep_modelcls)(self.event_encoder.input_size, hidden_size=hidden_size, **kwargs)

        if freeze_encoders:
            for param in self.parameters():
                param.requires_grad = False

        self.out_size = len(tables) * self.model.out_size
        self.tables = tables

    def forward(self, *inputs: Tensor):
        '''
        Input:
            input: Tensors for table with shape: N, L, C
        Output:
            Tensor of timestep embeddings with shape: N, L, C
        '''
        out = []
        measure_inds = {}
        for table, input in zip(self.tables, inputs):
            batch_size = input.size(0)
            input = input.reshape((-1, ) + input.shape[2:])  # NxL1, L2, C
            input = self.event_encoder(input)
            input, measure_ind = self.model(input)  # NxL, C
            measure_inds[table.table] = measure_ind

            input = input.reshape((batch_size, -1) + input.shape[1:])  # N, L, C
            out.append(input)

        return out, {'measure_inds': measure_inds}


class DeepTimestepEncoder(nn.Module):
    def __init__(self, joint_vocab, tables, hidden_size=50,
                 event_modelcls='models.EventEncoder',
                 timestep_modelcls='models.LinearMaxMeanSumPool', freeze_encoders=False,
                 pat_hidden_size=128,
                 value_vocab=None,
                 **kwargs):
        super().__init__()

        self.event_encoder = load_class(event_modelcls)(joint_vocab, value_vocab=value_vocab, **kwargs)
        self.model = load_class(timestep_modelcls)(self.event_encoder.input_size, hidden_size=hidden_size, **kwargs)
        self.mixer = nn.Sequential(nn.Linear(len(tables) * self.model.out_size, self.model.out_size),
                                   nn.ReLU(),
                                   nn.Linear(self.model.out_size, pat_hidden_size),
                                   nn.ReLU())

        for param in self.model.parameters():
            param.requires_grad = False

        if freeze_encoders:
            for param in self.parameters():
                param.requires_grad = False

        self.out_size = pat_hidden_size
        self.tables = tables

    def forward(self, *inputs: Tensor):
        '''
        Input:
            input: Tensors for table with shape: N, L, C
        Output:
            Tensor of timestep embeddings with shape: N, L, C
        '''
        out = []
        measure_inds = {}
        for table, input in zip(self.tables, inputs):
            batch_size = input.size(0)
            input = input.reshape((-1, ) + input.shape[2:])  # NxL1, L2, C
            input = self.event_encoder(input)
            input, measure_ind = self.model(input)  # NxL, C
            measure_inds[table.table] = measure_ind

            input = input.reshape((batch_size, -1) + input.shape[1:])  # N, L, C
            out.append(input)

        timesteps = torch.cat(out, -1)
        timesteps = self.mixer(timesteps)
        return timesteps, {'measure_inds': measure_inds}


class PatientDeepPoolEncoder(nn.Module):
    def __init__(self, input_size, tables, pat_padaware=False, timestep_modelcls='models.LinearMaxMeanSumPool', normalize='models.LNNormalizer', normalize_mlp=False, include_demfc=True, dem_size=8, dem_dropout=.0, include_timestep=False, visit_dropout=.0, freeze_encoders=False, **otherkw):
        super().__init__()

        if include_demfc:
            self.dem_fc = nn.Sequential(nn.Linear(dem_size, 40),
                                        nn.ReLU(),
                                        nn.Dropout(dem_dropout),
                                        nn.Linear(40, 20),
                                        nn.ReLU()
                                        )
            dem_size = 20

        linear_input_size = input_size + dem_size
        if include_timestep:
            linear_input_size += 2

        self.linear = nn.Linear(linear_input_size, input_size // len(tables))

        if freeze_encoders:
            for param in self.parameters():
                param.requires_grad = False

        if normalize:
            self.normalizers = nn.ModuleList(
                [load_class(normalize)(input_size, normalize_mlp) for _ in range(3)])


        self.include_demfc = include_demfc
        self.out_size = 3 * (input_size // len(tables)) + dem_size
        if include_timestep:
            self.out_size += 2
        self.normalize = normalize
        self.padaware = pat_padaware
        self.tables = tables
        self.include_timestep = include_timestep

    def forward(self, dem, *timesteps):
        '''
        Input:
            input: Tensor of timestep embeddings with shape: N, L, C
            dem: Tensor of demographics with shape: N, C
        Output:
            Tensor of patient embeddings for each timestep with shape: N, L, C
        '''

        N = timesteps[0].shape[0]
        L = timesteps[0].shape[1]

        if self.include_demfc:
            dem = self.dem_fc(dem)
            dem = dem.unsqueeze(1).expand(N, L, -1)  # N, L, C
            timesteps = list(timesteps) + [dem]

        if self.include_timestep:
            timestep = torch.arange(L, dtype=torch.float).to(dem.device).unsqueeze(1).unsqueeze(0).expand(N, -1, -1)  # N, L, C
            timesteps = list(timesteps) + [torch.log(timestep + 1), torch.exp(timestep/1000) - 1]

        timesteps = torch.cat(timesteps, -1)  # N, L, C'
        input = self.linear(timesteps)  # N, L, C

        input = F.pad(input.contiguous(), (0, 0, L-1, 0, 0, 0))  # N, L', C
        input = input.transpose(1, 2).contiguous()  # N, C, L

        p_max, p_max_ind = F.max_pool1d(input, L, 1, return_indices=True)  # N, C, L
        p_max_ind = (p_max_ind - (L - 1)).detach()
        # Collect max activations and indices
        # if hasattr(self, 'max_pool_size'):
        #     for i, table in enumerate(self.tables):
        #         time_inds[table.table] = p_max_ind[:, i*self.table_step_size: i*self.table_step_size + self.max_pool_size]
        #         activations[table.table] = p_max.detach()[:, i*self.table_step_size: i*self.table_step_size + self.max_pool_size]

        time_inds = p_max_ind
        activations = p_max.detach()

        p_max = p_max.transpose(1, 2).contiguous()  # N, L, C
        if self.normalize:
            p_max = self.normalizers[0](p_max)
        if self.padaware:
            p_avg = nonzero_avg_pool1d(input, L, 1).transpose(1, 2).contiguous()  # N, L, C
        else:
            p_avg = F.avg_pool1d(input, L, 1).transpose(1, 2).contiguous()  # N, L, C
        if self.normalize:
            p_avg = self.normalizers[1](p_avg)
        p_sum = norm_sum_pool1d(input, L, 1).transpose(1, 2).contiguous()  # N, L, C
        if self.normalize:
            p_sum = self.normalizers[2](p_sum)

        patient_timesteps = [p_max, p_avg, p_sum, dem]
        if self.include_timestep:
            timestep = torch.arange(L, dtype=torch.float).to(dem.device).unsqueeze(1).unsqueeze(0).expand(N, -1, -1)  # N, L, C
            patient_timesteps += [torch.log(timestep + 1), torch.exp(timestep/1000) - 1]

        patient_timesteps = torch.cat(patient_timesteps, -1)  # N, L, C'

        return patient_timesteps, {'time_inds': time_inds,
                                   'activations': activations}


class PatientPoolEncoder(nn.Module):
    def __init__(self, input_size, tables, pat_padaware=False, timestep_modelcls='models.LinearMaxMeanSumPool', normalize='models.LNNormalizer', normalize_mlp=False, include_demfc=True, dem_size=8, dem_dropout=.0, include_timestep=False, visit_dropout=.0, freeze_encoders=False, **otherkw):
        super().__init__()

        if include_demfc:
            self.dem_fc = nn.Sequential(nn.Linear(dem_size, 40),
                                        nn.ReLU(),
                                        nn.Dropout(dem_dropout),
                                        nn.Linear(40, 20),
                                        nn.ReLU()
                                        )
            dem_size = 20

        if freeze_encoders:
            for param in self.parameters():
                param.requires_grad = False

        if normalize:
            self.normalizers = nn.ModuleList(
                [load_class(normalize)(input_size, normalize_mlp) for _ in range(3)])

        self.include_demfc = include_demfc
        self.out_size = 3 * input_size + dem_size
        self.normalize = normalize
        self.padaware = pat_padaware
        self.tables = tables
        self.include_timestep = include_timestep
        if include_timestep:
            self.out_size += 2

    def forward(self, dem, *timesteps):
        '''
        Input:
            input: Tensor of timestep embeddings with shape: N, L, C
            dem: Tensor of demographics with shape: N, C
        Output:
            Tensor of patient embeddings for each timestep with shape: N, L, C
        '''

        patient_timesteps = []
        time_inds = {}
        activations = {}
        for i, input in enumerate(timesteps):
            N = input.shape[0]
            L = input.shape[1]

            input = F.pad(input.contiguous(), (0, 0, L-1, 0, 0, 0))  # N, L', C
            input = input.transpose(1, 2).contiguous()  # N, C, L

            p_max, p_max_ind = F.max_pool1d(input, L, 1, return_indices=True)  # N, C, L
            p_max_ind = (p_max_ind - (L - 1)).detach()
            # Collect max activations and indices
            # if hasattr(self, 'max_pool_size'):
            #     for i, table in enumerate(self.tables):
            #         time_inds[table.table] = p_max_ind[:, i*self.table_step_size: i*self.table_step_size + self.max_pool_size]
            #         activations[table.table] = p_max.detach()[:, i*self.table_step_size: i*self.table_step_size + self.max_pool_size]

            time_inds[self.tables[i].table] = p_max_ind
            activations[self.tables[i].table] = p_max.detach()

            p_max = p_max.transpose(1, 2).contiguous()  # N, L, C
            if self.normalize:
                p_max = self.normalizers[0](p_max)
            if self.padaware:
                p_avg = nonzero_avg_pool1d(input, L, 1).transpose(1, 2).contiguous()  # N, L, C
            else:
                p_avg = F.avg_pool1d(input, L, 1).transpose(1, 2).contiguous()  # N, L, C
            if self.normalize:
                p_avg = self.normalizers[1](p_avg)
            p_sum = norm_sum_pool1d(input, L, 1).transpose(1, 2).contiguous()  # N, L, C
            if self.normalize:
                p_sum = self.normalizers[2](p_sum)

            patient_timesteps += [p_max, p_avg, p_sum]

        if self.include_demfc:
            dem = self.dem_fc(dem)
        dem = dem.unsqueeze(1).expand(N, L, -1)  # N, L, C
        patient_timesteps.append(dem)

        if self.include_timestep:
            timestep = torch.arange(L, dtype=torch.float).to(dem.device).unsqueeze(1).unsqueeze(0).expand(N, -1, -1)  # N, L, C
            patient_timesteps += [torch.log(timestep + 1), torch.exp(timestep/1000) - 1]

        patient_timesteps = torch.cat(patient_timesteps, -1)  # N, L, C'

        return patient_timesteps, {'time_inds': time_inds,
                                   'activations': activations}


class PatientSmallChannelRNNEncoder(nn.Module):
    def __init__(self, input_size, tables, include_dem=True, include_demfc=True, dem_size=10, dem_dropout=.0, pat_hidden_size=128, visit_dropout=.0, pat_layers=1, freeze_encoders=False, **otherkw):
        super().__init__()

        input_size = input_size // len(tables)
        if include_dem:
            if include_demfc:
                self.dem_fc = nn.Sequential(nn.Linear(dem_size, 40),
                                            nn.ReLU(),
                                            nn.Dropout(dem_dropout),
                                            nn.Linear(40, 20),
                                            nn.ReLU()
                                            )
                dem_size = 20
            input_size += dem_size

        self.channel_rnns = nn.ModuleList([
            nn.GRU(input_size, input_size // 2, pat_layers, dropout=visit_dropout, batch_first=True)
            for _ in range(len(tables))])

        input_size = len(tables) * (input_size // 2)
        if include_dem:
            input_size += dem_size

        self.patient_rnn = nn.GRU(input_size, pat_hidden_size, pat_layers, dropout=visit_dropout, batch_first=True)

        if freeze_encoders:
            for param in self.parameters():
                param.requires_grad = False

        self.include_dem = include_dem
        self.include_demfc = include_demfc
        self.out_size = pat_hidden_size

    def forward(self, dem, *timesteps: Tensor):
        '''
        Input:
            input: Tensor of timestep embeddings with shape: N, L, C
            dem: Tensor of demographics with shape: N, C
        Output:
            Tensor of patient embeddings for each timestep with shape: N, L, C
        '''

        if self.include_dem:
            N = timesteps[0].shape[0]
            L = timesteps[0].shape[1]
            dem = self.dem_fc(dem)
            dem = dem.unsqueeze(1).expand(N, L, -1)  # N, L, C

        channels = []
        for channel_rnn, timestep in zip(self.channel_rnns, timesteps):
            if self.include_dem:
                timestep = torch.cat([timestep, dem], -1)  # N, L, C'

            timestep, _ = channel_rnn(timestep)
            channels.append(timestep)

        timesteps = torch.cat(channels, -1)

        if self.include_dem:
            timesteps = torch.cat([timesteps, dem], -1)  # N, L, C'

        timesteps, _ = self.patient_rnn(timesteps)
        return timesteps, {}


class PatientSharedChannelRNNEncoder(nn.Module):
    def __init__(self, input_size, tables, include_dem=True, include_demfc=True, dem_size=10, dem_dropout=.0, pat_hidden_size0=128, pat_hidden_size=128, visit_dropout=.0, pat_layers=1, freeze_encoders=False, **otherkw):
        super().__init__()

        input_size = input_size // len(tables)
        if include_dem:
            if include_demfc:
                self.dem_fc = nn.Sequential(nn.Linear(dem_size, 40),
                                            nn.ReLU(),
                                            nn.Dropout(dem_dropout),
                                            nn.Linear(40, 20),
                                            nn.ReLU()
                                            )
                dem_size = 20
            input_size += dem_size

        self.channel_rnn = nn.GRU(input_size, pat_hidden_size0, pat_layers, dropout=visit_dropout, batch_first=True)

        input_size = len(tables)*pat_hidden_size0
        if include_dem:
            input_size += dem_size

        self.patient_rnn = nn.GRU(input_size, pat_hidden_size, pat_layers, dropout=visit_dropout, batch_first=True)

        if freeze_encoders:
            for param in self.parameters():
                param.requires_grad = False

        self.include_dem = include_dem
        self.include_demfc = include_demfc
        self.out_size = pat_hidden_size

    def forward(self, dem, *timesteps: Tensor):
        '''
        Input:
            input: Tensor of timestep embeddings with shape: N, L, C
            dem: Tensor of demographics with shape: N, C
        Output:
            Tensor of patient embeddings for each timestep with shape: N, L, C
        '''

        if self.include_dem:
            N = timesteps[0].shape[0]
            L = timesteps[0].shape[1]
            dem = self.dem_fc(dem)
            dem = dem.unsqueeze(1).expand(N, L, -1)  # N, L, C

        channels = []
        for timestep in timesteps:
            if self.include_dem:
                timestep = torch.cat([timestep, dem], -1)  # N, L, C'

            timestep, _ = self.channel_rnn(timestep)
            channels.append(timestep)

        timesteps = torch.cat(channels, -1)

        if self.include_dem:
            timesteps = torch.cat([timesteps, dem], -1)  # N, L, C'

        timesteps, _ = self.patient_rnn(timesteps)
        return timesteps, {}


class PatientChannelRNNMaxPoolEncoder(nn.Module):
    def __init__(self, input_size, tables, include_dem=True, include_demfc=True, dem_size=10, dem_dropout=.0, pat_hidden_size0=128, pat_hidden_size=128, visit_dropout=.0, pat_layers=1, freeze_encoders=False, **otherkw):
        super().__init__()

        pool_input_size = input_size
        self.norm = nn.LayerNorm(pool_input_size)

        input_size = input_size // len(tables)
        if include_dem:
            if include_demfc:
                self.dem_fc = nn.Sequential(nn.Linear(dem_size, 40),
                                            nn.ReLU(),
                                            nn.Dropout(dem_dropout),
                                            nn.Linear(40, 20),
                                            nn.ReLU()
                                            )
                dem_size = 20
            input_size += dem_size

        self.channel_rnns = nn.ModuleList([
            nn.GRU(input_size, pat_hidden_size0, pat_layers, dropout=visit_dropout, batch_first=True)
            for _ in range(len(tables))])

        input_size = len(tables)*pat_hidden_size0
        if include_dem:
            input_size += dem_size

        self.patient_rnn = nn.GRU(input_size, pat_hidden_size, pat_layers, dropout=visit_dropout, batch_first=True)

        if freeze_encoders:
            for param in self.parameters():
                param.requires_grad = False

        self.tables = tables
        self.include_dem = include_dem
        self.include_demfc = include_demfc
        self.out_size = pat_hidden_size + pool_input_size

    def forward(self, dem, *timesteps: Tensor):
        '''
        Input:
            dem: Tensor of demographics with shape: N, C
            timesteps: Tensors of timestep embeddings with shape: N, L, C
        Output:
            Tensor of patient embeddings for each timestep with shape: N, L, C
        '''
        N = timesteps[0].shape[0]
        L = timesteps[0].shape[1]
        C = timesteps[0].shape[2]

        if self.include_dem:
            if self.include_demfc:
                dem = self.dem_fc(dem)
            dem = dem.unsqueeze(1).expand(N, L, -1)  # N, L, C

        timesteps_skip = torch.cat(timesteps, -1)
        timesteps_skip = F.pad(timesteps_skip.contiguous(), (0, 0, L - 1, 0, 0, 0))  # N, L', C
        timesteps_skip = timesteps_skip.transpose(1, 2).contiguous()  # N, C, L
        timesteps_skip, time_inds = F.max_pool1d(timesteps_skip, L, 1, return_indices=True)
        # remove padding
        time_inds = (time_inds - (L - 1)).detach()

        timesteps_skip = timesteps_skip.transpose(1, 2).contiguous()  # N, L, C
        timesteps_skip = self.norm(timesteps_skip)

        time_inds = {table.table: time_inds[:, i * C : (i + 1) * C].detach() for i, table in enumerate(self.tables)}
        activations = {table.table: timesteps_skip.transpose(1, 2)[:, i * C : (i + 1) * C].detach() for i, table in enumerate(self.tables)}

        channels = []
        for channel_rnn, timestep in zip(self.channel_rnns, timesteps):
            if self.include_dem:
                timestep = torch.cat([timestep, dem], -1)  # N, L, C'

            timestep, _ = channel_rnn(timestep)
            channels.append(timestep)

        patient_timesteps = torch.cat(channels, -1)

        if self.include_dem:
            patient_timesteps = torch.cat([patient_timesteps, dem], -1)  # N, L, C'

        patient_timesteps, _ = self.patient_rnn(patient_timesteps)

        patient_timesteps = torch.cat([timesteps_skip, patient_timesteps], -1)
        return patient_timesteps, {'time_inds': time_inds,
                                   'activations': activations}


class PatientChannelRNNEncoder(nn.Module):
    def __init__(self, input_size, tables, include_dem=True, include_demfc=True, dem_size=10, dem_dropout=.0, pat_hidden_size0=128, pat_hidden_size=128, visit_dropout=.0, pat_layers=1, freeze_encoders=False, **otherkw):
        super().__init__()

        input_size = input_size // len(tables)
        if include_dem:
            if include_demfc:
                self.dem_fc = nn.Sequential(nn.Linear(dem_size, 40),
                                            nn.ReLU(),
                                            nn.Dropout(dem_dropout),
                                            nn.Linear(40, 20),
                                            nn.ReLU()
                                            )
                dem_size = 20
            input_size += dem_size

        self.channel_rnns = nn.ModuleList([
            nn.GRU(input_size, pat_hidden_size0, pat_layers, dropout=visit_dropout, batch_first=True)
            for _ in range(len(tables))])

        input_size = len(tables)*pat_hidden_size0
        if include_dem:
            input_size += dem_size

        self.patient_rnn = nn.GRU(input_size, pat_hidden_size, pat_layers, dropout=visit_dropout, batch_first=True)

        if freeze_encoders:
            for param in self.parameters():
                param.requires_grad = False

        self.include_dem = include_dem
        self.include_demfc = include_demfc
        self.out_size = pat_hidden_size

    def forward(self, dem, *timesteps: Tensor):
        '''
        Input:
            input: Tensor of timestep embeddings with shape: N, L, C
            dem: Tensor of demographics with shape: N, C
        Output:
            Tensor of patient embeddings for each timestep with shape: N, L, C
        '''

        if self.include_dem:
            N = timesteps[0].shape[0]
            L = timesteps[0].shape[1]
            dem = self.dem_fc(dem)
            dem = dem.unsqueeze(1).expand(N, L, -1)  # N, L, C

        channels = []
        for channel_rnn, timestep in zip(self.channel_rnns, timesteps):
            if self.include_dem:
                timestep = torch.cat([timestep, dem], -1)  # N, L, C'

            timestep, _ = channel_rnn(timestep)
            channels.append(timestep)

        timesteps = torch.cat(channels, -1)

        if self.include_dem:
            timesteps = torch.cat([timesteps, dem], -1)  # N, L, C'

        timesteps, _ = self.patient_rnn(timesteps)
        return timesteps, {}


class PatientRNNEncoder(nn.Module):
    def __init__(self, input_size, include_dem=True, include_demfc=True, dem_size=10, dem_dropout=.0, pat_hidden_size0=128, pat_hidden_size=128, visit_dropout=.0, pat_layers=2, freeze_encoders=False, **otherkw):
        super().__init__()
        if include_dem:
            if include_demfc:
                self.dem_fc = nn.Sequential(nn.Linear(dem_size, 40),
                                            nn.ReLU(),
                                            nn.Dropout(dem_dropout),
                                            nn.Linear(40, 20),
                                            nn.ReLU()
                                            )
                dem_size = 20
            input_size += dem_size

        self.rnn = nn.GRU(input_size, pat_hidden_size, pat_layers, dropout=visit_dropout, batch_first=True)

        if freeze_encoders:
            for param in self.parameters():
                param.requires_grad = False

        self.include_dem = include_dem
        self.include_demfc = include_demfc
        self.out_size = pat_hidden_size + pool_skip_size

    def forward(self, dem, *timesteps):
        '''
        Input:
            input: Tensor of timestep embeddings with shape: N, L, C
            dem: Tensor of demographics with shape: N, C
        Output:
            Tensor of patient embeddings for each timestep with shape: N, L, C
        '''
        timesteps = torch.stack(timesteps, -1)
        N = timesteps.shape[0]
        L = timesteps.shape[1]

        timesteps_pool_skip = F.pad(timesteps.contiguous(), (0, 0, L-1, 0, 0, 0))  # N, L', C
        timesteps_pool_skip = timesteps_pool_skip.transpose(1, 2).contiguous()  # N, C, L
        timesteps_pool_skip, timesteps_pool_inds = F.max_pool1d(timesteps_pool_skip, L, 1, return_indices=True)
        timesteps_pool_skip = timesteps_pool_skip.transpose(1, 2).contiguous()  # N, L, C

        if self.include_dem:
            if self.include_demfc:
                dem = self.dem_fc(dem)
            dem = dem.unsqueeze(1).expand(N, L, -1)  # N, L, C

            timesteps = torch.cat([timesteps, dem], -1)  # N, L, C'

        timesteps, _ = self.rnn(timesteps)

        timesteps = torch.cat([timesteps_pool_skip, timesteps], -1)
        return timesteps, {'timesteps_pool_ind': timesteps_pool_inds}


class PatientRNNMaxPoolEncoder(nn.Module):
    def __init__(self, input_size, tables, include_dem=True, include_demfc=True, dem_size=10, dem_dropout=.0, pat_hidden_size=128, visit_dropout=.0, pat_layers=2, freeze_encoders=False, **otherkw):
        super().__init__()
        pool_skip_size = input_size
        if include_dem:
            if include_demfc:
                self.dem_fc = nn.Sequential(nn.Linear(dem_size, 40),
                                            nn.ReLU(),
                                            nn.Dropout(dem_dropout),
                                            nn.Linear(40, 20),
                                            nn.ReLU()
                                            )
                dem_size = 20
            input_size += dem_size

        self.norm = nn.LayerNorm(pool_skip_size)
        self.rnn = nn.GRU(input_size, pat_hidden_size, pat_layers, dropout=visit_dropout, batch_first=True)

        if freeze_encoders:
            for param in self.parameters():
                param.requires_grad = False

        self.tables = tables
        self.include_dem = include_dem
        self.include_demfc = include_demfc
        self.out_size = pat_hidden_size + pool_skip_size

    def forward(self, dem, *timesteps):
        '''
        Input:
            input: Tensor of timestep embeddings with shape: N, L, C
            dem: Tensor of demographics with shape: N, C
        Output:
            Tensor of patient embeddings for each timestep with shape: N, L, C
        '''
        timesteps = torch.cat(timesteps, -1)
        N = timesteps.shape[0]
        L = timesteps.shape[1]

        timesteps_pool_skip = F.pad(timesteps.contiguous(), (0, 0, L-1, 0, 0, 0))  # N, L', C
        timesteps_pool_skip = timesteps_pool_skip.transpose(1, 2).contiguous()  # N, C, L
        timesteps_pool_skip, time_inds = F.max_pool1d(timesteps_pool_skip, L, 1, return_indices=True)
        time_inds = (time_inds - (L - 1)).detach()
        timesteps_pool_skip_norm = timesteps_pool_skip.transpose(1, 2).contiguous()  # N, L, C
        timesteps_pool_skip_norm = self.norm(timesteps_pool_skip_norm)

        C = timesteps[0].shape[-1]
        time_inds = {table.table: time_inds[:, i * C : (i + 1) * C] for i, table in enumerate(self.tables)}
        activations = {table.table: timesteps_pool_skip[:, i * C : (i + 1) * C] for i, table in enumerate(self.tables)}

        if self.include_dem:
            if self.include_demfc:
                dem = self.dem_fc(dem)
            dem = dem.unsqueeze(1).expand(N, L, -1)  # N, L, C

            timesteps = torch.cat([timesteps, dem], -1)  # N, L, C'

        timesteps, _ = self.rnn(timesteps)

        timesteps = torch.cat([timesteps_pool_skip_norm, timesteps], -1)
        return timesteps, {'time_inds': time_inds,
                           'activations': activations}


class PatientMeanEncoder(nn.Module):
    def __init__(self, input_size, include_demfc=True, dem_size=10, dem_dropout=.0, pat_hidden_size=128, visit_dropout=.0, pat_layers=1, freeze_encoders=False, **otherkw):
        super().__init__()
        if include_demfc:
            self.dem_fc = nn.Sequential(nn.Linear(dem_size, 40),
                                        nn.ReLU(),
                                        nn.Dropout(dem_dropout),
                                        nn.Linear(40, 20),
                                        nn.ReLU()
                                        )
            dem_size = 20

        input_size += dem_size

        if freeze_encoders:
            for param in self.parameters():
                param.requires_grad = False

        self.include_demfc = include_demfc
        self.out_size = input_size

    def forward(self, dem, *timesteps):
        '''
        Input:
            input: Tensor of timestep embeddings with shape: N, L, C
            dem: Tensor of demographics with shape: N, C
        Output:
            Tensor of patient embeddings for each timestep with shape: N, L, C
        '''
        timesteps = torch.cat(out, -1)

        N = timesteps.shape[0]
        L = timesteps.shape[1]
        if self.include_demfc:
            dem = self.dem_fc(dem)
        dem = dem.unsqueeze(1).expand(N, L, -1)  # N, L, C

        timestep_dem = torch.cat([timesteps, dem], -1)  # N, L, C'

        L = timestep_dem.shape[1]

        input = F.pad(timestep_dem.contiguous(), (0, 0, L-1, 0))  # N, L', C
        input = input.transpose(1, 2).contiguous()  # N, C, L
        p_avg = nonzero_avg_pool1d(input, L, 1)
        p_avg = p_avg.transpose(1, 2).contiguous()  # N, L, C

        # out = torch.cat([timestep_dem, p_avg], dim=2)

        return F.relu(p_avg), {}


class PatientCNNEncoder(nn.Module):
    def __init__(self, input_size, include_demfc=True, dem_size=10, dem_dropout=.0, pat_hidden_size=128, visit_dropout=.0, pat_layers=1, freeze_encoders=False, **otherkw):
        super().__init__()
        if include_demfc:
            self.dem_fc = nn.Sequential(nn.Linear(dem_size, 40),
                                        nn.ReLU(),
                                        nn.Dropout(dem_dropout),
                                        nn.Linear(40, 20),
                                        nn.ReLU()
                                        )
            dem_size = 20

        input_size += dem_size
        self.cnn = DeepCNN(input_size, pat_hidden_size, visit_dropout)

        if freeze_encoders:
            for param in self.parameters():
                param.requires_grad = False

        self.include_demfc = include_demfc
        self.out_size = pat_hidden_size * 2

    def forward(self, dem, *timesteps):
        '''
        Input:
            input: Tensor of timestep embeddings with shape: N, L, C
            dem: Tensor of demographics with shape: N, C
        Output:
            Tensor of patient embeddings for each timestep with shape: N, L, C
        '''

        timesteps = torch.cat(out, -1)

        N = timesteps.shape[0]
        L = timesteps.shape[1]
        if self.include_demfc:
            dem = self.dem_fc(dem)
        dem = dem.unsqueeze(1).expand(N, L, -1)  # N, L, C

        timestep_dem = torch.cat([timesteps, dem], -1)  # N, L, C'

        timestep_dem = self.cnn(timestep_dem.transpose(1,2).contiguous()).transpose(1,2).contiguous()

        # L = timestep_dem.shape[1]

        # input = F.pad(timestep_dem.contiguous(), (0, 0, L-1, 0, 0, 0))  # N, L', C
        # input = input.transpose(1, 2).contiguous()  # N, C, L
        # p_avg = nonzero_avg_pool1d(input, L, 1)
        # p_avg = p_avg.transpose(1, 2).contiguous()  # N, L, C

        # out = torch.cat([timestep_dem, p_avg], dim=2)

        return F.relu(timestep_dem), {}


class PatientRETAINEncoder(nn.Module):
    def __init__(self, input_size, include_dem=True, include_demfc=True, dem_size=10, dem_dropout=.0, pat_hidden_size=128, visit_dropout=.0, pat_layers=1, freeze_encoders=False, **otherkw):
        super().__init__()
        if include_dem:
            if include_demfc:
                self.dem_fc = nn.Sequential(nn.Linear(dem_size, 40),
                                            nn.ReLU(),
                                            nn.Dropout(dem_dropout),
                                            nn.Linear(40, 20),
                                            nn.ReLU()
                                            )
                dem_size = 20
            input_size += dem_size

        # Visit-level attention
        self.visit_rnn = nn.GRU(input_size, pat_hidden_size, pat_layers, dropout=visit_dropout, batch_first=True)
        self.visit_attention = nn.Linear(pat_hidden_size, 1)

        # Variable-level attention
        self.variable_rnn = nn.GRU(input_size, pat_hidden_size, pat_layers, dropout=visit_dropout, batch_first=True)
        self.variable_attention = nn.Linear(pat_hidden_size, input_size)

        if freeze_encoders:
            for param in self.parameters():
                param.requires_grad = False

        self.include_demfc = include_demfc
        self.include_dem = include_dem
        self.out_size = input_size

    def forward(self, dem, *timesteps):
        '''
        Input:
            input: Tensor of timestep embeddings with shape: N, L, C
            dem: Tensor of demographics with shape: N, C
        Output:
            Tensor of patient embeddings for each timestep with shape: N, L, C
        '''
        if self.include_dem:
            N = timesteps[0].shape[0]
            L = timesteps[0].shape[1]
            if self.include_demfc:
                dem = self.dem_fc(dem)
            dem = dem.unsqueeze(1).expand(N, L, -1)  # N, L, C

            timesteps = torch.cat(list(timesteps) + [dem], -1)  # N, L, C'

        alphas, _ = self.visit_rnn(timesteps)
        alphas = self.visit_attention(alphas)
        alphas = F.tanh(alphas)

        betas, _ = self.variable_rnn(timesteps)
        betas = self.variable_attention(betas)
        betas = F.tanh(betas)

        timesteps = timesteps * alphas * betas

        L = timesteps.shape[1]
        input = F.pad(timesteps.contiguous(), (0, 0, L-1, 0))  # N, L', C
        input = input.transpose(1, 2).contiguous()  # N, C, L
        out = F.max_pool1d(input, L, 1)
        # out = norm_sum_pool1d(input, L, 1)
        out = out.transpose(1, 2).contiguous()  # N, L, C

        return out, {}
