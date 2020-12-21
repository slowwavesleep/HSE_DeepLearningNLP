import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

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

    @staticmethod
    def mask_pads(weights, x_len, max_len):
        mask = torch.arange(max_len)[None, :] < x_len[:, None]
        masked_weights = weights.clone()
        masked_weights[~mask] = float('-inf')
        return masked_weights

    def forward(self, x, y, x_len, max_len):

        query = self.query_projection(y)  # (batch_size, seq_len, dim)
        key = self.key_projection(x)  # (batch_size, seq_len, dim)
        value = self.value_projection(x)  # (batch_size, seq_len, dim)

        attention_weights = torch.bmm(query, key.permute(0, 2, 1))  # (batch_size, seq_len, seq_len)
        attention_weights /= self.scale_factor

        attention_weights = self.mask_pads(attention_weights,
                                           x_len=x_len,
                                           max_len = max_len)

        attention_weights = torch.softmax(attention_weights, dim=1)  # (batch_size, seq_len, seq_len)

        attention = torch.bmm(attention_weights, value)  # (batch_size, seq_len, dim)

        return y + attention


class MyNet(nn.Module):

    def __init__(self,
                 dim: int,
                 hidden_size: int,
                 vocab_size: int,
                 dropout: float,
                 max_len: int = 64,
                 pad_index: int = 0,
                 weight_tying=True):

        super().__init__()

        self.dim = dim
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.pad_index = pad_index
        self.weight_tying = weight_tying

        self.embedding_layer = nn.Embedding(num_embeddings=self.vocab_size,
                                            embedding_dim=self.dim,
                                            padding_idx=self.pad_index)

        self.embedding_dropout = SpatialDropout(p=dropout)

        self.attention_layer = AttentionLayer(self.dim)

        self.lstm_1 = nn.LSTM(self.dim,
                              self.hidden_size,
                              batch_first=True,
                              bidirectional=False)

        self.lstm_2 = nn.LSTM(self.dim,
                              self.hidden_size,
                              batch_first=True,
                              bidirectional=False)

        self.final_output = nn.Linear(in_features=self.hidden_size,
                                      out_features=self.vocab_size)

        if self.weight_tying and self.dim == self.hidden_size:
            self.final_output.weight = self.embedding_layer.weight

    @staticmethod
    def count_pads(x, axis=1):
        try:
            x = x.cpu()
        except:
            x = x
        return torch.Tensor(np.count_nonzero(x, axis=axis))

    def forward(self, x, y):

        x_lengths = self.count_pads(x)
        y_lengths = self.count_pads(y)

        x = self.embedding_layer(x)
        x = self.embedding_dropout(x)
        y = self.embedding_layer(y)
        y = self.embedding_dropout(y)

        x = pack_padded_sequence(x,
                                 x_lengths,
                                 batch_first=True,
                                 enforce_sorted=False)

        y = pack_padded_sequence(y,
                                 y_lengths,
                                 batch_first=True,
                                 enforce_sorted=False)

        x, memory = self.lstm_1(x)
        y, _ = self.lstm_2(y, memory)

        x = pad_packed_sequence(x,
                                batch_first=True,
                                total_length=self.max_len)[0]
        y = pad_packed_sequence(y,
                                batch_first=True,
                                total_length=self.max_len)[0]

        y = self.attention_layer(x,
                                 y,
                                 x_len=x_lengths,
                                 max_len=self.max_len)

        y = self.final_output(y)

        return y
