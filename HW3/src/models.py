import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from torch import Tensor
from typing import Union
from torch.nn import functional as F


def scaled_dot_product_attention(query: Tensor,
                                 key: Tensor,
                                 value: Tensor,
                                 mask: Union[None, Tensor] = None) -> Tensor:

    similarity = query.bmm(key.transpose(1, 2))

    scale = query.size(-1) ** 0.5

    if mask is not None:
        similarity = similarity.masked_fill(mask, float('-inf'))

    softmax = F.softmax(similarity / scale, dim=-1)

    return softmax.bmm(value)


def get_non_pad_lens(seq):
    lens = seq.size(-1) - (seq == 0).sum(-1)
    return lens


def get_pad_mask(seq_1, seq_2):
    # (batch_size, seq_len_1), (batch_size, seq_len_2)  -> (batch_size, seq_len_2, seq_len_1)
    seq_len_1 = seq_1.size(-1)
    seq_len_2 = seq_2.size(-1)
    lens = get_non_pad_lens(seq_1)
    masks = [torch.arange(seq_len_1).expand(seq_len_2, seq_len_1) >= true_len for true_len in lens]
    return torch.stack(masks).cuda()


class SpatialDropout(torch.nn.Dropout2d):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        x = x.unsqueeze(2)  # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T)
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x


class AttentionLayer(nn.Module):

    def __init__(self,
                 dim: int):
        super().__init__()

        self.dim = dim

        self.key_projection = nn.Linear(in_features=self.dim,
                                        out_features=self.dim)
        self.value_projection = nn.Linear(in_features=self.dim,
                                          out_features=self.dim)
        self.query_projection = nn.Linear(in_features=self.dim,
                                          out_features=self.dim)

        self.scale_factor = np.sqrt(self.dim)

    def forward(self, encoder_seq, decoder_seq, mask):

        query = self.query_projection(decoder_seq)
        key = self.key_projection(encoder_seq)
        value = self.value_projection(encoder_seq)

        attention = scaled_dot_product_attention(query, key, value, mask)

        return attention


class MyNet(nn.Module):

    def __init__(self,
                 emb_dim: int,
                 hidden_size: int,
                 vocab_size: int,
                 dropout: float,
                 pad_index: int = 0,
                 weight_tying: bool = True):

        super().__init__()

        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=emb_dim,
                                            padding_idx=pad_index)

        self.embedding_dropout = SpatialDropout(p=dropout)

        self.attention_layer = AttentionLayer(emb_dim)

        self.lstm_1 = nn.LSTM(input_size=emb_dim,
                              hidden_size=hidden_size,
                              batch_first=True,
                              bidirectional=False)

        self.lstm_2 = nn.LSTM(input_size=emb_dim,
                              hidden_size=hidden_size,
                              batch_first=True,
                              bidirectional=False)

        self.final_output = nn.Linear(in_features=hidden_size,
                                      out_features=vocab_size)

        if weight_tying and emb_dim == hidden_size:
            self.final_output.weight = self.embedding_layer.weight

        self.layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, encoder_seq, decoder_seq):

        mask = get_pad_mask(encoder_seq, decoder_seq)

        enc_lengths = get_non_pad_lens(encoder_seq)
        dec_lengths = get_non_pad_lens(decoder_seq)

        enc_initial_len = encoder_seq.size(-1)
        dec_initial_len = decoder_seq.size(-1)

        encoder_seq = self.embedding_layer(encoder_seq)
        encoder_seq = self.embedding_dropout(encoder_seq)
        encoder_seq = self.layer_norm(encoder_seq)

        decoder_seq = self.embedding_layer(decoder_seq)
        decoder_seq = self.embedding_dropout(decoder_seq)
        decoder_seq = self.layer_norm(decoder_seq)

        encoder_seq = pack_padded_sequence(input=encoder_seq,
                                           lengths=enc_lengths,
                                           batch_first=True,
                                           enforce_sorted=False)

        encoder_seq, memory = self.lstm_1(encoder_seq)

        encoder_seq = pad_packed_sequence(sequence=encoder_seq,
                                          batch_first=True,
                                          total_length=enc_initial_len)[0]

        decoder_seq = pack_padded_sequence(input=decoder_seq,
                                           lengths=dec_lengths,
                                           batch_first=True,
                                           enforce_sorted=False)

        decoder_seq, _ = self.lstm_2(decoder_seq, memory)

        decoder_seq = pad_packed_sequence(sequence=decoder_seq,
                                          batch_first=True,
                                          total_length=dec_initial_len)[0]

        attention = self.attention_layer(encoder_seq, decoder_seq, mask)

        return self.final_output(decoder_seq + attention)


