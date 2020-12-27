import math

import torch
from torch import nn
from models.modules import nonzero_avg_pool


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, ninp, nhead=4, pat_hidden_size=128, pat_layers=3, visit_dropout=0.5, dem_size=10, include_demfc=True, dem_dropout=0.3, **kwargs):
        super().__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer

        if include_demfc:
            self.dem_fc = nn.Sequential(nn.Linear(dem_size, 40),
                                        nn.ReLU(),
                                        nn.Dropout(dem_dropout),
                                        nn.Linear(40, 20),
                                        nn.ReLU()
                                        )
            dem_size = 20

        ninp += dem_size

        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, visit_dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, pat_hidden_size, visit_dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, pat_layers)
        self.ninp = ninp

        self.out_size = ninp
        self.include_demfc = include_demfc

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, dem):
        '''
        Input
            src: N, L, C
            dem: N, C
        Output
            N, L, C
        '''
        src = src.transpose(0, 1).contiguous()  # L, N, C

        # mask out for auto-regressiveness
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask  # L, L

        L = src.shape[0]
        N = src.shape[1]
        if self.include_demfc:
            dem = self.dem_fc(dem)
        dem = dem.unsqueeze(1).expand(L, N, -1)  # L, N, C
        src = torch.cat([src, dem], -1)  # L, N, C'

        src = self.pos_encoder(src * math.sqrt(self.ninp))
        output = self.transformer_encoder(src, self.src_mask)

        output = output.transpose(0, 1).contiguous()
        return output, {}


class TransformerTimestepModel(nn.Module):
    def __init__(self, ninp, nhead=3, hidden_size=128, input_dropout=0.5, layers=2, **kwargs):
        super().__init__()
        self.model_type = 'Transformer'
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, hidden_size, input_dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, layers)

        self.ninp = ninp
        self.out_size = ninp

    def forward(self, input):
        '''
        Input: N, L, C
        Output: N, C
        '''
        mask = (input[:, :, :100].detach() == 0).all(2).contiguous()  # N, L
        src = input.transpose(0, 1).contiguous()  # L, N, C

        # Mask out paddings
        output = self.transformer_encoder(
            src,
            src_key_padding_mask=mask
            )

        breakpoint()
        output = output.transpose(0, 1).contiguous()  # N, L, C
        output = nonzero_avg_pool(output, input, dim=1)  # N, C

        output = output.masked_fill(torch.isnan(output), 0).squeeze(-1)
        return output, {}
