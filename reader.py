import itertools
import pandas as pd
import copy 

class DatasetReader(object):

    def __init__(self, dataset_file, dataset_format):
        print(dataset_file)
        self.dataset_file = dataset_file
        self.format = dataset_format
        self.dataset = []
        self.test = []
        self.character_count = 0
        self.original_dataset = []

    def trim(self, index, n_chunks):
        step = len(self.dataset) // n_chunks
        self.original_dataset = copy.deepcopy(self.dataset)
        chunks = [ self.dataset[i:i+step] for i in range(0, len(self.dataset), step)]
        self.dataset = chunks[index]
        
    def read(self):
        def error_handling():
            self.default()
        getattr(self,self.format,error_handling)()
        self.character_count = sum([len(x) for x in self.dataset])

    def default(self):
        self.BOUN()

    def BOUN(self):
        with open(self.dataset_file,'r') as f:
            data = f.read().split('\n')
        data = [ x.strip() for x in data ]
        self.test = data[::]
        data = [ x.replace(" ","") for x in data ]
        self.dataset = data[::]

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