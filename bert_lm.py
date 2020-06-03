import numpy as np
import torch
from transformers import (
    BertForMaskedLM,
    BertTokenizer
)

class BertLM(object):

    def __init__(self, model_name_or_path, device='cuda', gpu_batch_size=None):
        self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
        self.model = BertForMaskedLM.from_pretrained(model_name_or_path)
        self.device = device
        self.model.to(device, non_blocking=True)
        self.model.eval()

    def batch_encode(self,list_of_strings):
        batch_encoding = [ self.tokenizer.encode_plus(item) for item in list_of_strings ]

        input_ids = np.array([x['input_ids'] for x in batch_encoding ])
        attention_mask = np.array([x['attention_mask'] for x in batch_encoding ])

        input_ids = self.insert_padding(input_ids, 0)
        attention_mask = self.insert_padding(attention_mask, 0)
        
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        
        return input_ids, attention_mask
    
    def calculate_loss(self,input_ids, attention_mask=None):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        outputs = self.model(input_ids=input_ids, masked_lm_labels=input_ids, attention_mask=attention_mask, encoder_attention_mask=attention_mask)
        for idx, item in enumerate(attention_mask):
            current_mask = item.view(1,-1).transpose(0,1)
            outputs[1][idx] = current_mask * outputs[1][idx]
        probs = []
        for idx, hypothesis in enumerate(outputs[1]):
            ppl = 0.0
            for idx2, token in enumerate(hypothesis):
                if not len(token[token.nonzero()]):
                    continue
                ppl += token[input_ids[idx][idx2]].item()
                
            hypothesis_length = len([ x for x in hypothesis if len(x[x.nonzero()]) ])
            if hypothesis_length:
                ppl = np.float(torch.abs(torch.sum(item.view(-1))))
                ppl = ppl / hypothesis_length
            else:
                ppl = 0.0
            probs.append(1 / ppl)

        return probs
    
    def insert_padding(self,arr,padding):
        max_len = np.max([len(a) for a in arr])
        padded_arr = np.asarray([np.pad(a, (0, max_len - len(a)), 'constant', constant_values=padding) for a in arr])
        return padded_arr

    def get_probs(self, list_of_lists):
        input_ids, attention_mask = self.batch_encode(list_of_lists)
        probs = self.calculate_loss(input_ids, attention_mask)
        return probs