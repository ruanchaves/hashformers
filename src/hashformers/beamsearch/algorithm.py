import itertools
import re
from hashformers.beamsearch.data_structures import (
    Node,
    ProbabilityDictionary
)

from hashformers.beamsearch.model_lm import ModelLM

class Beamsearch(ModelLM):

    def __init__(
            self,
            model_name_or_path=None, 
            model_type=None, 
            device='cuda', 
            gpu_batch_size=1):
        super().__init__(
            model_name_or_path=model_name_or_path, 
            model_type=model_type, 
            device=device, 
            gpu_batch_size=gpu_batch_size)

    def next_step(self, list_of_candidates):
        output = []
        for candidate_string in list_of_candidates:
            candidates = [ 
                candidate_string[:pos] + ' ' + candidate_string[pos:] \
                    if pos else candidate_string for pos in range(len(candidate_string)) 
                ]
            candidates = list(filter(lambda x: not re.findall(".*?(?=\s{2})", x), candidates))
            output.extend(candidates)
        return output

    def update_probabilities(self, tree, prob_dict):
        for item in tree:
            current_batch = []
            for word in item:
                if word in prob_dict:
                    continue
                else:
                    current_batch.append(word)
            if current_batch:
                current_batch_probs = self.model.get_probs(current_batch)
            for idx, word in enumerate(current_batch):
                prob_dict[word] = current_batch_probs[idx]
        return prob_dict

    def reshape_tree(self, tree, measure):
        return [ tree[x:x+measure] for x in range(0, len(tree), measure) ]

    def flatten_list(self, list_):
        return [ item for sublist in list_ for item in sublist ]

    def trim_tree(self, tree, prob_dict, topk):
        output = []
        probs = [ prob_dict[x] for x in tree ]
        candidates = [
            Node(item, item.replace(" ", ""), probs[idx]) for idx, item in enumerate(tree)
        ]
        for key, group in itertools.groupby(candidates, key=lambda x: x.characters):
            sorted_group = sorted(list(group), key=lambda x: x.score)
            trimmed_group = sorted_group[0:topk]
            trimmed_group = [x.hypothesis for x in trimmed_group]
            output.extend(trimmed_group)
        return output

    def run(self, dataset, topk=20, steps=13):
        tree = dataset
        prob_dict = {}
        for i in range(steps):
            tree = self.next_step(tree)
            tree = self.reshape_tree(tree, self.gpu_batch_size)
            prob_dict = self.update_probabilities(tree, prob_dict)
            tree = self.flatten_list(tree)
            tree = self.trim_tree(tree, prob_dict, topk)
        return ProbabilityDictionary(prob_dict)