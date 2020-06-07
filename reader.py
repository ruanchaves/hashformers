import itertools
import pandas as pd
import copy 

class DatasetReader(object):

    def __init__(self, dataset_file, dataset_format):
        self.dataset_file = dataset_file
        self.format = dataset_format
        self.dataset = []
        self.test = []
        
    def read(self):
        def error_handling():
            self.default()
        getattr(self,self.format,error_handling)()

    def get_character_count(self):
        return sum([len(x) for x in self.dataset])

    def default(self):
        self.BOUN()

    def BOUN(self):
        with open(self.dataset_file,'r') as f:
            data = f.read().split('\n')

        self.test = [ x.strip() for x in data ]
        self.dataset = [ x.replace(" ","") for x in data ]

    def trim(self, n_chunks=-1, index=-1):
        def trim_dataset(dataset, n_chunks, index):
            step = len(dataset) // n_chunks
            chunks = [ dataset[i:i+step] for i in range(0, len(dataset), step)]
            dataset = chunks[index]
            return dataset

        if n_chunks != -1 and index != -1:
            pass
            self.dataset = trim_dataset(self.dataset, n_chunks, index)
            self.test = trim_dataset(self.test, n_chunks, index)

    def glushkova(self):
        data = pd.read_csv(self.dataset_file)
        self.dataset = data['hashtag'].astype(str).values.tolist()
        data['test'] = data['hashtag'].combine(data['true_segmentation'], self.labels_to_tokens)
        self.test = data['test'].astype(str).values.tolist()

    def labels_to_tokens(self, tokens, labels):
        tokens = [ x for x in tokens ] 
        labels = [ None if x == 0 else ' ' for x in labels ]
        new_tokens = list(itertools.chain(*zip(tokens, labels)))
        new_tokens = list(filter(lambda x: x is not None, new_tokens))
        new_tokens = "".join(new_tokens)
        return new_tokens