from lm_scorer.models.auto import GPT2LMScorer as LMScorer
import numpy as np
import torch
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer
)

class GPT2LM(object):

    def __init__(self, model_name_or_path, device='cuda', gpu_batch_size=20, gpu_expansion_batch_size=2):
        self.scorer = LMScorer(model_name_or_path, device=device, batch_size=gpu_batch_size)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path, pad_token_id=self.tokenizer.eos_token_id)
        self.device = device
        self.expansions = []
        self.gpu_expansion_batch_size = gpu_expansion_batch_size
        self.model.to(self.device)

    def group_batches_by_length(self, arr):
        
        def length_check(len_batch, len_sorted_arr, idx):
            if (len_batch == self.gpu_expansion_batch_size) or (len_sorted_arr - 1 == idx):
                return True
            else:
                return False
        
        def get_batch_len(batch):
            try:
                return len(batch[-1])
            except IndexError:
                return -1

        len_array = [ len(x) for x in arr ]
        sorted_arr = [x for _,x in sorted(zip(len_array,arr))]
        output = []
        batch = []
        for idx, item in enumerate(sorted_arr):
            if (not batch) or (get_batch_len(batch) == len(item)):
                batch.append(item)
                if length_check(len(batch), len(sorted_arr), idx):
                    output.append(batch)
                    batch = []
            else:
                output.append(batch)
                batch = []
                batch.append(item)
                if length_check(len(batch), len(sorted_arr), idx):
                    output.append(batch)
                    batch = []
        return output

    def sort_generations_by_input_sequence(self, sentences, generations):
        sorted_output = []
        for sentence in sentences:
            for generation in generations:
                if generation.find(sentence) == 0:
                    sorted_output.append({'hypothesis': sentence, 'generation': generation})
        return sorted_output

    def generate_expansions(self, 
                            sentences,  
                            skip_special_tokens=True, 
                            do_sample=True,
                            max_length=30,
                            top_k=50,
                            top_p=0.95,
                            num_return_sequences=10):
        input_ids = [ self.tokenizer.encode(x) for x in sentences ]
        input_ids_batches = self.group_batches_by_length(input_ids)
        result = []
        for batch in input_ids_batches:
            batch_tensor = torch.tensor(batch)
            batch_tensor = batch_tensor.to(self.device)
            greedy_output = self.model.generate(
                batch_tensor,
                do_sample=do_sample,
                max_length=max_length, 
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_return_sequences
            )

            batch_result = [ self.tokenizer.decode(greedy_output[i], skip_special_tokens=skip_special_tokens) for i in range(len(greedy_output))]

            result.extend(batch_result)
        result = self.sort_generations_by_input_sequence(sentences, result)
        self.expansions.append(result)

    # def get_best_candidate(self, list_of_candidates, search='greedy', expansions=10, **kwargs):
    #     generations = self.generate_expansions(list_of_candidates, **kwargs)
    #     # generation_list = [ x['generation'] for x in greedy_generations ]
    #     # candidate_generations = self.get_token_probs(generation_list)
    #     # for idx, item in enumerate(greedy_generations):
    #     #     greedy_generations[idx]['tokens'] = candidate_generations[idx]
    #     #     return greedy_generations
    #     raise NotImplementedError

    def get_probs(self, list_of_candidates):
        scores =  self.scorer.sentence_score(list_of_candidates, log=True)
        scores = [ 1-x for x in scores ]
        return scores
    
    def get_token_probs(self, list_of_candidates):
        scores =  self.scorer.tokens_score(list_of_candidates)
        return scores

def main():
    gpt2 = GPT2LM('gpt2')
    gpt2.generate_expansions(['AlQa eda', 'Al Qaeda'])
    print(gpt2.expansions)


if __name__ == '__main__':
    main()
