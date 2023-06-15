import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Embedding(nn.Module):
    def __init__(self, vocab_size, dimension):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dimension, padding_idx=0)
        self.dimension = dimension

    def forward(self, input_vec):
        return self.embedding(input_vec) * math.sqrt(self.dimension)


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, dimension, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        positional_enc = torch.zeros(max_seq_len, dimension)

        den = torch.pow(
            10000, torch.div(torch.arange(0, dimension / 2) * 2, float(dimension))
        )
        num = torch.arange(0, max_seq_len).unsqueeze(1)

        positional_enc[:, 0::2], positional_enc[:, 1::2] = (
            torch.sin(num / den),
            torch.cos(num / den),
        )
        positional_enc = positional_enc.unsqueeze(0)
        self.register_buffer("positional_enc", positional_enc)

    def forward(self, input_vec):
        seq_len = input_vec.size(1)
        return self.dropout(input_vec + Variable(self.positional_enc[:, :seq_len]))


class MultiHeadedAttention(nn.Module):
    def __init__(self, dimension, heads, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.dimension = dimension
        self.queryl = nn.Linear(dimension, dimension)
        self.keyl = nn.Linear(dimension, dimension)
        self.valuel = nn.Linear(dimension, dimension)
        self.outl = nn.Linear(dimension, dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):

        assert self.dimension == query.size(-1)
        batch_size = query.size(0)

        query = self.queryl(query)
        key = self.keyl(key)
        value = self.valuel(value)

        query = query.view(
            batch_size, -1, self.heads, query.size(-1) // self.heads
        ).transpose(1, 2)
        key = key.view(
            batch_size, -1, self.heads, key.size(-1) // self.heads
        ).transpose(1, 2)

        value = value.view(
            batch_size, -1, self.heads, value.size(-1) // self.heads
        ).transpose(1, 2)

        attn = self.attention(query, key, value, mask, self.dropout)

        concat = attn.transpose(1, 2).reshape(
            batch_size, -1, query.size(-1) * self.heads
        )

        return self.outl(concat)

    def attention(self, query, key, value, mask=None, dropout=None):
        qk = torch.div(
            torch.matmul(query, key.transpose(-2, -1)), math.sqrt(query.size(-1))
        )

        if mask is not None:
            mask = mask.unsqueeze(1)
            qk = qk.masked_fill(mask == 0, -1e9)

        qk = nn.Softmax(dim=-1)(qk)
        qk = self.dropout(qk) if dropout is not None else qk
        return torch.matmul(qk, value)


class FeedForwardNet(nn.Module):
    def __init__(self, dimension, dff=2048, dropout=0.1):
        super().__init__()
        self.l = nn.Linear(dimension, dff)
        self.out = nn.Linear(dff, dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_vec):
        return self.out(self.dropout(F.relu(self.l(input_vec))))


class LayerNorm(nn.Module):
    def __init__(self, dimension, delta=1e-6):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(dimension))
        self.bias = nn.Parameter(torch.zeros(dimension))
        self.delta = delta

    def forward(self, input_vec):
        mean = torch.mean(input_vec, dim=-1, keepdim=True)
        std = torch.std(input_vec, dim=-1, keepdim=True) + self.delta
        return (self.gain / std) * (input_vec - mean) + self.bias


class ResidualConnection(nn.Module):
    def __init__(self, dimension, dropout=0.1):
        super().__init__()
        self.norm = LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_vec, sublayer):
        return input_vec + self.dropout(sublayer(self.norm(input_vec)))


class EncoderLayer(nn.Module):
    def __init__(self, dimension, head=8, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadedAttention(dimension, head, dropout)
        self.ffnn = FeedForwardNet(dimension, dropout=dropout)
        self.resconn1 = ResidualConnection(dimension, dropout)
        self.resconn2 = ResidualConnection(dimension, dropout)

        self.norm = LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_vec, mask=None):
        attn = self.resconn1(input_vec, lambda x: self.attn(x, x, x, mask))
        return self.resconn2(attn, self.ffnn), attn


class Encoder(nn.Module):
    def __init__(
        self, vocab_size, number_of_layers, head, max_seq_len, dimension, dropout
    ):
        super().__init__()
        self.emb = Embedding(vocab_size, dimension)
        self.penc = PositionalEncoding(max_seq_len, dimension, dropout)
        self.enclays = nn.ModuleList(
            [
                copy.deepcopy(EncoderLayer(dimension, head, dropout))
                for _ in range(number_of_layers)
            ]
        )
        self.norm = LayerNorm(dimension)

    def forward(self, input_vec, mask=None):
        # input_vec - batch_size, max_sent_len
        emb = self.emb(input_vec)  # emb size = batch, max_sent_len, dimension
        emb = self.penc(emb)

        for layer in self.enclays:
            emb, _ = layer(emb, mask)
        emb = self.norm(emb)
        return emb, _


class DecoderLayer(nn.Module):
    def __init__(self, dimension, heads=8, dropout=0.1):
        super().__init__()
        self.ffnn = FeedForwardNet(dimension, dropout=dropout)
        self.resconn = nn.ModuleList(
            [copy.deepcopy(ResidualConnection(dimension, dropout)) for _ in range(3)]
        )
        self.attn = nn.ModuleList(
            [
                copy.deepcopy(MultiHeadedAttention(dimension, heads, dropout))
                for _ in range(2)
            ]
        )

    def forward(self, input_vec, encoder_output, encmask, decmask):
        selfattn = self.resconn[0](input_vec, lambda x: self.attn[0](x, x, x, decmask))

        encdecattn = self.resconn[1](
            selfattn, lambda x: self.attn[1](x, encoder_output, encoder_output, encmask)
        )

        return self.resconn[2](encdecattn, self.ffnn)


class Decoder(nn.Module):
    def __init__(
        self, vocab_size, number_of_layers, head, max_seq_len, dimension, dropout
    ):
        super().__init__()
        self.emb = Embedding(vocab_size, dimension)
        self.penc = PositionalEncoding(max_seq_len, dimension, dropout)
        self.declays = nn.ModuleList(
            [
                copy.deepcopy(DecoderLayer(dimension, head, dropout))
                for i in range(number_of_layers)
            ]
        )
        self.norm = LayerNorm(dimension)

    def forward(self, input_vec, encoder_output, encmask, decmask):

        emb = self.emb(input_vec)
        emb = self.penc(emb)

        for layer in self.declays:
            emb = layer(emb, encoder_output, encmask, decmask)
        return self.norm(emb)


class Transformer(nn.Module):
    def __init__(
        self,
        envocab_size,
        devocab_size,
        enc_max_seq_len,
        dec_max_seq_len,
        head,
        number_of_enc_layers,
        number_of_dec_layers,
        dimension,
        dropout,
    ):
        super().__init__()
        self.encoder = Encoder(
            envocab_size,
            number_of_enc_layers,
            head,
            enc_max_seq_len,
            dimension,
            dropout,
        )
        self.decoder = Decoder(
            devocab_size,
            number_of_dec_layers,
            head,
            dec_max_seq_len,
            dimension,
            dropout,
        )
        self.ffnn = nn.Linear(dimension, devocab_size)

    def forward(self, enc_input_vec, dec_input_vec, encmask, decmask):

        encout, _ = self.encoder(enc_input_vec, encmask)
        decout = self.decoder(dec_input_vec, encout, encmask, decmask)

        return F.log_softmax(self.ffnn(decout), dim=-1)


class Batch:
    def __init__(self, src, trg=None, device="cpu", pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(1)

        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.ntokens = (self.trg_y != pad).data.sum()

            trg_mask = (self.trg != pad).unsqueeze(-2)
            dimension = self.trg.size(-1)

            mask = torch.tril(torch.ones(1, dimension, dimension, device=device))
            mask[mask != 0] = 1
            mask = Variable(mask > 0)

            if self.trg.is_cuda:
                mask.cuda()
            self.trg_mask = trg_mask & mask


class LabelSmoothing(nn.Module):
    def __init__(self, vocab_size, pad_index, alpha):
        super().__init__()
        self.alpha = alpha
        self.vocab_size = vocab_size
        self.pad_index = pad_index

    def forward(self, prediction, target):
        prediction = prediction.contiguous().view(-1, prediction.size(-1))
        target = target.contiguous().view(-1)

        one_hot_target = torch.nn.functional.one_hot(
            target, num_classes=prediction.size(-1)
        )
        one_hot_target[:, self.pad_index] = 0
        one_hot_target = (one_hot_target * (1 - self.alpha)) + (
            self.alpha / (self.vocab_size - 2)
        )
        one_hot_target.masked_fill_((target == self.pad_index).unsqueeze(1), 0)

        return F.kl_div(prediction, one_hot_target, reduction="sum")


class CustomAdam:
    def __init__(self, dimension, optimizer, warmup_steps=4000, step_num=0):
        self.optimizer = optimizer
        self.step_num = step_num
        self.dimension = dimension
        self.warmup_steps = warmup_steps

    def step(self):
        self.step_num += 1
        lr = self.rate()

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        self.optimizer.step()

    def rate(self):
        return self.dimension ** (-0.5) * min(
            self.step_num ** (-0.5), self.step_num * self.warmup_steps ** (-1.5)
        )


def init_weights(m):
    if hasattr(m, "weight") and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
