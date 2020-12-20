from tqdm import tqdm
import zipfile
from typing import Tuple, Dict, Iterable, List
import numpy as np
from torch.utils.data import Dataset
import torch
from nltk.tokenize import wordpunct_tokenize


def load_embeddings(zip_path: str,
                    filename: str,
                    pad_token: str = '<PAD>',
                    unk_token: str = '<UNK>',
                    max_words: int = 100_000,
                    verbose: bool = True) -> Tuple[Dict[str, int], np.ndarray]:

    vocab = dict()
    embeddings = list()

    with zipfile.ZipFile(zip_path) as zipped_file:
        with zipped_file.open(filename) as file_object:

            vocab_size, embedding_dim = file_object.readline().decode('utf-8').strip().split()

            vocab_size = int(vocab_size)
            embedding_dim = int(embedding_dim)

            max_words = vocab_size if max_words <= 0 else max_words

            vocab[pad_token] = len(vocab)
            vocab[unk_token] = len(vocab)
            embeddings.append(np.zeros(embedding_dim))
            embeddings.append(np.random.normal(0, 0.15, size=embedding_dim))

            progress_bar = tqdm(total=max_words, disable=not verbose)

            for line in file_object:
                parts = line.decode('utf-8').strip().split()

                token = ' '.join(parts[:-embedding_dim]).lower()

                if token in vocab:
                    continue

                word_vector = np.array(list(map(float, parts[-embedding_dim:])))

                vocab[token] = len(vocab)
                embeddings.append(word_vector)

                progress_bar.update()

                if len(vocab) == max_words:
                    break

            progress_bar.close()

    embeddings = np.stack(embeddings)

    return vocab, embeddings


class TextClassificationDataset(Dataset):

    def __init__(self,
                 texts: Iterable[str],
                 targets: Iterable[int],
                 vocab: dict,
                 pre_pad: bool = False,
                 pad_index: int = 0,
                 unk_index: int = 1,
                 max_length: int = 32):

        super().__init__()

        self.texts = texts
        self.targets = targets
        self.vocab = vocab

        self.unk_index = unk_index
        self.pre_pad = pre_pad
        self.pad_index = pad_index
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    @staticmethod
    def filter_unk(token_indices: List[int], unk_token: int = 1) -> List[int]:
        output = []
        unk_flag = False
        for token in token_indices:
            if token != unk_token:
                output.append(token)
            elif token == unk_token and not unk_flag:
                unk_flag = True
                output.append(token)
            else:
                pass
        return output

    def tokenize(self, text: str) -> List[int]:

        tokens = wordpunct_tokenize(text)

        token_indices = [self.vocab.get(token, 1) for token in tokens]

        token_indices = self.filter_unk(token_indices, self.unk_index)

        return token_indices

    def padding(self, tokenized_text: List[int]) -> List[int]:

        tokenized_text = tokenized_text[:self.max_length]

        if self.pre_pad:
            tokenized_text = [self.pad_index] * (self.max_length - len(tokenized_text)) + tokenized_text
        else:
            tokenized_text += [self.pad_index] * (self.max_length - len(tokenized_text))

        return tokenized_text

    def __getitem__(self, index: int) -> Tuple[torch.tensor, int]:

        text = self.texts[index]
        target = self.targets[index]

        tokenized_text = self.tokenize(text)
        tokenized_text = self.padding(tokenized_text)

        tokenized_text = torch.tensor(tokenized_text)

        return tokenized_text, target
