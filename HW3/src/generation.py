import torch
from torch.nn import Module
import youtokentome as yttm
from typing import List, NoReturn
from tqdm import tqdm
from scipy.special import softmax
import numpy as np
from torch.nn import functional as F


class Generator:

    def __init__(self,
                 tokenizer: yttm.BPE,
                 model: Module,
                 device: object,
                 bos_index: int = 2,
                 eos_index: int = 3,
                 max_sequence: int = 32):

        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.bos_index = bos_index
        self.eos_index = eos_index
        self.max_sequence = max_sequence

    def greedy_search(self, source: str) -> str:

        tokenized = self.tokenizer.encode(source, eos=True, bos=True)

        encoder_sequence = torch.tensor([tokenized]).long().to(self.device)
        decoder_sequence = torch.tensor([self.bos_index]).long().unsqueeze(0).to(self.device)

        self.model.eval()

        with torch.no_grad():
            for timestamp in range(self.max_sequence):
                predictions = self.model(encoder_sequence, decoder_sequence)
                current_token = predictions[:, -1, :].argmax(dim=-1)
                if current_token == self.eos_index:
                    break
                decoder_sequence = torch.cat([decoder_sequence, current_token.unsqueeze(0)], dim=-1)

        response = self.tokenizer.decode(decoder_sequence.squeeze(0).tolist())
        response = response[0].lstrip('<BOS> ').rstrip('<EOS>')

        return response

    def nucleus_search(self,
                       question,
                       p=0.92):

        tokenized = self.tokenizer.encode(question, eos=True, bos=True)

        tokenized = self.tokenizer.encode(question, eos=True, bos=True)

        encoder_sequence = torch.tensor([tokenized]).long().to(self.device)
        decoder_sequence = torch.tensor([self.bos_index]).long().unsqueeze(0).to(self.device)

        self.model.eval()

        with torch.no_grad():
            for timestamp in range(self.max_sequence):
                # по идее это и есть условная вероятность следующего слова:
                predictions = torch.softmax(self.model(encoder_sequence, decoder_sequence), dim=-1)
                candidate_probs, candidate_tokens = predictions[:, -1, :].sort(dim=-1, descending=True)
                # ищем индекс, левее которого вероятности складываются в `p`
                candidate_probs = torch.cumsum(candidate_probs, dim=-1)
                new_candidate_probs = candidate_probs[candidate_probs < p]
                # если получили пустой тензор, то просто используем все вероятности
                if new_candidate_probs.nelement() == 0:
                    new_candidate_probs = candidate_probs.squeeze(0)
                # еще один softmax, чтобы выбранные вероятности снова складывались в единицу
                candidate_probs = softmax(new_candidate_probs.cpu().numpy())
                candidate_tokens = candidate_tokens[:, :candidate_probs.shape[0]].squeeze(0).cpu().numpy()
                current_token = np.random.choice(candidate_tokens, p=candidate_probs)
                if current_token == self.eos_index:
                    break
                current_token = torch.Tensor([current_token]).long().to(self.device)
                decoder_sequence = torch.cat([decoder_sequence, current_token.unsqueeze(0)], dim=-1)

        answer = self.tokenizer.decode(decoder_sequence.squeeze(0).tolist())
        answer = answer[0].lstrip('<BOS>').rstrip('<EOS>')
        return answer

    def to_file(self,
                source_sentences: List[str],
                target_sentences: List[str],
                file_path: str,
                kind: str = 'greedy') -> NoReturn:

        with open(file_path, 'w') as file:
            for source, target in tqdm(zip(source_sentences, target_sentences),
                                       total=len(source_sentences),
                                       desc=f'Generating responses using {kind} search...'):

                if kind == 'greedy':
                    response = self.greedy_search(source)
                elif kind == 'nucleus':
                    response = self.nucleus_search(source)
                file.write(source + '\n')
                file.write(target + '\n')
                file.write(response + '\n')
                file.write('\n')
