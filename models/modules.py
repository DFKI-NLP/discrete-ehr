import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from utils import load_class


def min_pool1d(x, kernel_size, stride=1):
    ''' x: N, C, L
    '''
    x = x.unfold(2, kernel_size, stride)
    x = x.min(dim=-1)[0]
    return x


def min_pool(x):
    ''' x: N, C, L
    '''
    x = x.min(dim=-1)[0]
    return x


def nonzero_avg_pool1d(x, kernel_size, stride=1):
    x = x.unfold(2, kernel_size, stride)
    div = (x.detach() != 0).sum(dim=-1).float()
    x = x.sum(dim=-1) / (div + 1e-5)
    return x


def nonzero_avg_pool(x, inp, dim=-1):
    '''
    0-Padding aware average pooling
    Input:
    x: [N, C, L]
    inp: [N, L, C]
    Output: [N, C, 1]
    '''
    div = (inp.detach()[:, :, :100] != 0).all(2).sum(dim=1, keepdim=True).float().detach()
    x = x.sum(dim=dim) / (div + 1e-5)
    return x[:, :, None]


def norm_sum_pool(x, dim=-1):
    norm = torch.sqrt(torch.abs(x.sum(dim=dim, keepdim=True)))
    x = x.sum(dim=dim, keepdim=True) / (norm + 1e-5)
    return x


def norm_sum_pool1d(x, kernel_size, stride=1):
    x = x.unfold(2, kernel_size, stride)
    norm = torch.sqrt(torch.abs(x.sum(dim=-1)))
    x = x.sum(dim=-1) / (norm + 1e-5)
    return x


class LNNormalizer(nn.Module):
    def __init__(self, input_size, normalize_mlp):
        super().__init__()
        layers = [nn.LayerNorm(input_size)]
        if normalize_mlp:
            layers.extend([
                nn.Linear(input_size, 200),
                nn.ReLU(),
                nn.Linear(200, input_size),
                nn.ReLU(),
                nn.LayerNorm(input_size)
                ])
        self.normalizer = nn.Sequential(*layers, nn.ReLU())

    def forward(self, input):
        '''
        Input and Output: shape N, C
        '''
        return self.normalizer(input)


class LNWORNormalizer(nn.Module):
    def __init__(self, input_size, normalize_mlp):
        super().__init__()
        layers = [nn.LayerNorm(input_size)]
        if normalize_mlp:
            layers.extend([
                nn.Linear(input_size, 200),
                nn.ReLU(),
                nn.Linear(200, input_size),
                nn.ReLU(),
                nn.LayerNorm(input_size)
                ])
        self.normalizer = nn.Sequential(*layers)

    def forward(self, input):
        '''
        Input and Output: shape N, C
        '''
        return self.normalizer(input)


class Normalizer(nn.Module):
    def __init__(self, input_size, normalize_mlp, normalize_p=1):
        super().__init__()
        self.normalize_p = normalize_p
        layers = []
        if normalize_mlp:
            layers.extend([
                nn.Linear(input_size, 200),
                nn.ReLU(),
                nn.Linear(200, input_size),
                nn.ReLU(),
                ])
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        '''
        Input and Output: shape N, C
        '''
        input = F.normalize(input, self.normalize_p, dim=1)
        input = self.model(input)
        return input


class DeepCNN(nn.Module):
    def __init__(self, input_size, hidden_size, drop_p, kernel_sizes=[[1, 4, 12], [1, 4]]):
        super().__init__()
        self.hidden_size = hidden_size

        convs = []
        in_c = input_size
        for kernel_size in kernel_sizes:
            convs.append(nn.ModuleList([nn.Conv1d(in_channels=in_c,
                                                  out_channels=hidden_size,
                                                  kernel_size=k) for k in kernel_size]))
            in_c = hidden_size*len(kernel_size)

        self.convs = nn.ModuleList(convs)
        self.dropout = nn.Dropout(drop_p)

    def forward(self, input):
        out = input
        for i, conv in enumerate(self.convs, 1):
            out = [F.relu(conv_(F.pad(out, (conv_.weight.shape[-1]-1, 0)))) for conv_ in conv]

            last_conv = i == len(self.convs)

            # out = [F.max_pool1d(c, 3 if not last_conv else c.size(-1), 2) for c in out]
            out = torch.cat(out, dim=1)

        out = self.dropout(out)
        return out


class BenchmarkPredictor(nn.Module):
    def __init__(self, input_size, decision_dropout, **otherkw):
        super().__init__()
        self.decision_mlps = nn.ModuleDict(dict(in_hospital_mortality=nn.Linear(input_size, 1),
                                                length_of_stay_classification=nn.Linear(input_size, 10),
                                                length_of_stay_regression=nn.Linear(input_size, 1),
                                                phenotyping=nn.Linear(input_size, 25),
                                                decompensation=nn.Linear(input_size, 1)))
        self.decision_dropout = nn.Dropout(decision_dropout)

    def forward(self, input):
        '''
        Input:
            Tensor of patient embeddings for each timestep with shape: N, L, C
        Output:
            Dictionary of predictions
        '''
        input = self.decision_dropout(input)

        last_timestep = input[:, -1]
        if input.shape[1] > 47:
            ihm_timestep = input[:, 47]
        else:
            ihm_timestep = input[:, 0]

        preds = {}
        preds['phenotyping'] = self.decision_mlps['phenotyping'](last_timestep)  # N, 25
        preds['in_hospital_mortality'] = self.decision_mlps['in_hospital_mortality'](ihm_timestep)  # N, 1
        preds['length_of_stay_classification'] = self.decision_mlps['length_of_stay_classification'](input)  # N, L, 10
        preds['length_of_stay_regression'] = self.decision_mlps['length_of_stay_regression'](input).squeeze(-1)  # N, L
        preds['decompensation'] = self.decision_mlps['decompensation'](input).squeeze(-1)  # N, L

        return preds, {}


class SumPool(nn.Module):
    def __init__(self, input_size, input_dropout, **otherkws):
        super().__init__()
        self.input_dropout = nn.Dropout(input_dropout)
        self.d = input_size
        self.out_size = input_size

    def forward(self, input):
        input = self.input_dropout(input)
        out = F.relu(input).transpose(1, 2)
        o_sum = norm_sum_pool(out).squeeze(-1)
        return o_sum, None


class MaxMeanSumPool(nn.Module):
    def __init__(self, input_size, input_dropout, **otherkws):
        super().__init__()
        self.input_dropout = nn.Dropout(input_dropout)
        self.d = input_size
        self.out_size = input_size * 3

    def forward(self, input):
        input = self.input_dropout(input)
        out = F.relu(input).transpose(1, 2)
        o_max, max_inds = F.max_pool1d(out, out.size(-1), return_indices=True)
        # o_avg = nonzero_avg_pool(out, input)
        o_avg = F.avg_pool1d(out, out.size(-1))
        o_sum = norm_sum_pool(out)
        out = torch.cat([o_max, o_avg, o_sum], 1).squeeze(-1)
        return out, max_inds


class LinearMaxMeanSumPool(nn.Module):
    def __init__(self, input_size, hidden_size, input_dropout=0.0, padaware=False, include_eventd=False, normalize='models.Normalizer', normalize_mlp=False, **otherkws):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_dropout = nn.Dropout(input_dropout)
        self.linear = nn.Linear(input_size, hidden_size)
        self.d = hidden_size
        self.out_size = hidden_size * 3

        if normalize:
            self.normalizers = nn.ModuleList(
                [load_class(normalize)(hidden_size, normalize_mlp) for _ in range(3)])

        self.padaware = padaware
        self.normalize = normalize
        self.include_eventd = include_eventd
        if include_eventd:
            self.out_size += 1

    def forward(self, input):
        '''
        Input:
            input: T, L, C
        '''
        input = self.input_dropout(input)  # T, L, C
        out = F.relu(self.linear(input))  # T, L, C
        if self.padaware:
            mask = (input[:, :, :100] == 0).all(2)[:, :, None]
            out = out.masked_fill_(mask, 0)
        out = out.transpose(1, 2).contiguous()  # T, C, L
        o_max, max_inds = F.max_pool1d(out, out.size(-1), return_indices=True)
        o_max = o_max.squeeze(-1)
        if self.normalize:
            o_max = self.normalizers[0](o_max)
        if self.padaware:
            o_avg = nonzero_avg_pool(out, input).squeeze(-1)
        else:
            o_avg = F.avg_pool1d(out, out.size(-1)).squeeze(-1)
        if self.normalize:
            o_avg = self.normalizers[1](o_avg)
        o_sum = norm_sum_pool(out).squeeze(-1)
        if self.normalize:
            o_sum = self.normalizers[2](o_sum)
        outs = [o_max, o_avg, o_sum]

        if self.include_eventd:
            event_counts = (input[:, :, :100].detach() != 0).all(2).sum(1, keepdim=True).float() / input.size(1)  # T
            outs.append(event_counts)
        out = torch.cat(outs, 1)  # T, C
        return out, max_inds


class LinearMaxMeanSumMinPool(nn.Module):
    def __init__(self, input_size, hidden_size, input_dropout=0.0, padaware=False, include_eventd=False, normalize='models.Normalizer', normalize_mlp=False, **otherkws):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_dropout = nn.Dropout(input_dropout)
        self.linear = nn.Linear(input_size, hidden_size)
        self.d = hidden_size
        self.out_size = hidden_size * 4

        if normalize:
            self.normalizers = nn.ModuleList(
                [load_class(normalize)(hidden_size, normalize_mlp) for _ in range(4)])

        self.padaware = padaware
        self.normalize = normalize
        self.include_eventd = include_eventd
        if include_eventd:
            self.out_size += 1

    def forward(self, input):
        '''
        Input:
            input: T, L, C
        '''
        input = self.input_dropout(input)  # T, L, C
        out = F.relu(self.linear(input))  # T, L, C
        if self.padaware:
            mask = (input[:, :, :100].detach() != 0).all(2).byte()
            out = (out * mask[:, :, None])
        out = out.transpose(1, 2).contiguous()  # T, C, L

        o_max, max_ind = F.max_pool1d(out, out.size(-1), return_indices=True)
        o_max = o_max.squeeze(-1)
        if self.normalize:
            o_max = self.normalizers[0](o_max)

        if self.padaware:
            o_avg = nonzero_avg_pool(out, input).squeeze(-1)
        else:
            o_avg = F.avg_pool1d(out, out.size(-1)).squeeze(-1)
        if self.normalize:
            o_avg = self.normalizers[1](o_avg)

        o_sum = norm_sum_pool(out).squeeze(-1)
        if self.normalize:
            o_sum = self.normalizers[2](o_sum)

        o_min = norm_sum_pool(out).squeeze(-1)
        if self.normalize:
            o_min = self.normalizers[3](o_min)

        outs = [o_max, o_avg, o_sum, o_min]

        if self.include_eventd:
            event_counts = (input[:, :, :100].detach() != 0).all(2).sum(1, keepdim=True).float() / input.size(1)  # T
            outs.append(event_counts)
        out = torch.cat(outs, 1)  # T, C
        return out, max_ind


class DeepMaxMeanSumPool(nn.Module):
    def __init__(self, input_size, hidden_size, input_dropout=0.0, padaware=False, normalize='models.Normalizer', normalize_mlp=False, **otherkws):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_dropout = nn.Dropout(input_dropout)
        self.linear = nn.Sequential(nn.Linear(input_size, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, hidden_size))
        self.d = hidden_size
        self.out_size = hidden_size * 3

        if normalize:
            self.normalizers = nn.ModuleList(
                [load_class(normalize)(hidden_size, normalize_mlp) for _ in range(3)])

        self.padaware = padaware
        self.normalize = normalize

    def forward(self, input):
        input = self.input_dropout(input)
        out = F.relu(self.linear(input))
        if self.padaware:
            mask = (input[:, :,:100].detach() != 0).all(2).byte()
            out = out.masked_fill_(mask, 0)
        out = out.transpose(1, 2)
        o_max, max_inds = F.max_pool1d(out, out.size(-1), return_indices=True)
        o_max = o_max.squeeze(-1)
        if self.normalize:
            o_max = self.normalizers[0](o_max)
        if self.padaware:
            o_avg = nonzero_avg_pool(out, input).squeeze(-1)
        else:
            o_avg = F.avg_pool1d(out, out.size(-1)).squeeze(-1)
        if self.normalize:
            o_avg = self.normalizers[1](o_avg)
        o_sum = norm_sum_pool(out).squeeze(-1)
        if self.normalize:
            o_sum = self.normalizers[2](o_sum)
        out = torch.cat([o_max, o_avg, o_sum], 1)
        return out, max_inds


class LinearMaxSumNormPool(nn.Module):
    def __init__(self, input_size, hidden_size, input_dropout, **otherkws):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_dropout = nn.Dropout(input_dropout)
        self.linear = nn.Linear(input_size, hidden_size)
        self.d = hidden_size
        self.out_size = hidden_size * 2
        self.norm = nn.LayerNorm(self.out_size)

    def forward(self, input):
        input = self.input_dropout(input)
        out = F.relu(self.linear(input)).transpose(1, 2)
        o_max, max_inds = F.max_pool1d(out, out.size(-1), return_indices=True)
        # o_avg = nonzero_avg_pool(out, input)
        o_sum = norm_sum_pool(out)
        out = torch.cat([o_max, o_sum], 1).squeeze(-1)
        out = self.norm(out)
        return out, max_inds



class LinearMaxSumPool(nn.Module):
    def __init__(self, input_size, hidden_size, input_dropout, **otherkws):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_dropout = nn.Dropout(input_dropout)
        self.linear = nn.Linear(input_size, hidden_size)
        self.d = hidden_size
        self.out_size = hidden_size * 2

    def forward(self, input):
        input = self.input_dropout(input)
        out = F.relu(self.linear(input)).transpose(1, 2)
        o_max, max_inds = F.max_pool1d(out, out.size(-1), return_indices=True)
        # o_avg = nonzero_avg_pool(out, input)
        o_sum = norm_sum_pool(out)
        out = torch.cat([o_max, o_sum], 1).squeeze(-1)
        return out, max_inds


class LinearMaxPadMeanSumPool(nn.Module):
    def __init__(self, input_size, hidden_size, input_dropout, **otherkws):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_dropout = nn.Dropout(input_dropout)
        self.linear = nn.Linear(input_size, hidden_size)
        self.d = hidden_size
        self.out_size = hidden_size * 3

    def forward(self, input):
        input = self.input_dropout(input)
        out = F.relu(self.linear(input)).transpose(1, 2)
        o_max, max_inds = F.max_pool1d(out, out.size(-1), return_indices=True)
        o_avg = nonzero_avg_pool(out, input)
        o_sum = norm_sum_pool(out)
        out = torch.cat([o_max, o_avg, o_sum], 1).squeeze(-1)
        return out, max_inds


class LinearSumNormPool(nn.Module):
    def __init__(self, input_size, hidden_size, input_dropout, **otherkws):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_dropout = nn.Dropout(input_dropout)
        self.linear = nn.Linear(input_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.out_size = hidden_size

    def forward(self, input):
        input = self.input_dropout(input)
        out = F.relu(self.linear(input)).transpose(1, 2)
        out = norm_sum_pool(out).squeeze(-1)
        out = self.norm(out)
        return out, None


class LinearSumPool(nn.Module):
    def __init__(self, input_size, hidden_size, input_dropout, **otherkws):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_dropout = nn.Dropout(input_dropout)
        self.linear = nn.Linear(input_size, hidden_size)
        self.out_size = hidden_size

    def forward(self, input):
        input = self.input_dropout(input)
        out = F.relu(self.linear(input)).transpose(1, 2)
        out = norm_sum_pool(out)
        return out.squeeze(-1), None


class LinearMaxPool(nn.Module):
    def __init__(self, input_size, hidden_size, input_dropout, **otherkws):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_dropout = nn.Dropout(input_dropout)
        self.linear = nn.Linear(input_size, hidden_size)
        self.out_size = hidden_size

    def forward(self, input):
        input = self.input_dropout(input)
        out = F.relu(self.linear(input)).transpose(1, 2)
        out, max_inds = F.max_pool1d(out, out.size(-1), return_indices=True)
        return out.squeeze(-1), max_inds
