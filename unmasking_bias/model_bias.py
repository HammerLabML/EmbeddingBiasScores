import torch
from transformers import AutoModelForMaskedLM, PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
import difflib
import numpy as np

from typing import Tuple, List


""" This is a super class with functions to determine internal biases (in terms of token probabilities) of MLMs.
The idea of measuring bias based on MLM preferences for token replacement relates to the following papers:
TODO
"""


class MLMBiasDataset(torch.utils.data.Dataset):

    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {'input_ids': self.encodings.input_ids[idx].clone().detach(),
                'label': self.encodings.label[idx].clone().detach(),
                'mask_ids': self.encodings.mask_ids[idx],
                'attention_mask': self.encodings.attention_mask[idx]}

    def __len__(self):
        return len(self.encodings.input_ids)


class MLMBiasTester:

    def __init__(self, mlm: AutoModelForMaskedLM, tokenizer: PreTrainedTokenizer, batch_size: int):
        self.softmax = torch.nn.Softmax(dim=2)
        self.mlm = mlm
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.use_cuda = torch.cuda.is_available()

        self.special_tokens_ids = [tokenizer.cls_token_id, tokenizer.eos_token_id, tokenizer.bos_token,
                                   tokenizer.sep_token_id, tokenizer.pad_token_id, tokenizer.unk_token_id,
                                   tokenizer.mask_token_id] + tokenizer.additional_special_tokens_ids

        if self.use_cuda:
            self.mlm = self.mlm.to('cuda')
            self.softmax = self.softmax.to('cuda')
            print('Using torch with CUDA/GPU')
        else:
            print('WARNING! Using torch on CPU!')

    def is_mask(self, value):
        return value == self.tokenizer.mask_token_id

    def get_batch_token_probabilities(self, batch: BatchEncoding) -> List[float]:
        tokens = batch['label']
        tokens_masked = batch['input_ids']
        mask_ids = batch['mask_ids']
        attention_mask = batch['attention_mask']

        # determine mask token ids here
        #mask = tokens_masked.clone()
        #mask.apply_(is_mask)

        #mask_ids2 = mask.argwhere().tolist()  # this is a list of elements [sample_id, token_id]
        #print(mask_ids2)

        if self.use_cuda:
            tokens = tokens.to('cuda')
            tokens_masked = tokens_masked.to('cuda')
            attention_mask = attention_mask.to('cuda')

        output = self.mlm(tokens_masked, attention_mask=attention_mask)
        hidden_states = output.logits
        token_prob = self.softmax(hidden_states)

        sel_token_probs = [token_prob[i][mask_ids[i]].tolist() for i in range(len(mask_ids))]

        target_token_ids = [tokens[i][mask_id] for i, mask_id in enumerate(mask_ids)]
        log_probs_target = [float(sel_token_probs[i][target_token_id]) for i, target_token_id in enumerate(target_token_ids)]

        # gpu cleanup
        if self.use_cuda:
            hidden_states = hidden_states.to('cpu')
            token_prob = token_prob.to('cpu')
            tokens.to('cpu')
            tokens_masked.to('cpu')

            del output
            del hidden_states
            del token_prob

            torch.cuda.empty_cache()

        return log_probs_target

    def get_token_probabilities(self, dataset: MLMBiasDataset) -> List[float]:
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        res = []
        for batch_id, sample in enumerate(loader):
            res += self.get_batch_token_probabilities(sample)
        return res

