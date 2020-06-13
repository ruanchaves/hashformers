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

    def stanford(self):
        with open(self.dataset_file,'r') as f:
            dataset_text = f.read().split('\n')
            dataset_text = [ x.split('\t') for x in dataset_text]
            dataset_text = dataset_text[1:-1]
        data = pd.DataFrame(dataset_text)
        data = data.rename(columns={
            0: 'tweet_id',
            1: 'hashtag',
            2: 'segmentation',
            3: 'label'
        })
        
        assert(data.shape == data.dropna().shape)

        data['label'] = data['label'].astype(int)
        data['hashtag'] = data['hashtag'].str.strip("'")
        data['segmentation'] = data['segmentation'].str.strip("'")
        data = data[data['label']==1]
        self.test = data['segmentation'].values.tolist()
        self.dataset = data['hashtag'].values.tolist()

    def labels_to_tokens(self, tokens, labels):
        tokens = [ x for x in tokens ] 
        labels = [ None if int(x) == 0 else ' ' for x in labels ]
        new_tokens = list(itertools.chain(*zip(tokens, labels)))
        new_tokens = list(filter(lambda x: x is not None, new_tokens))
        new_tokens = "".join(new_tokens).strip()
        return new_tokens

def main():
    reader = DatasetReader('/run/media/user/DADOS/NLP/datasets/hashtag_segmentation/glushkova/test_eng.csv', 'glushkova')
    reader.read()
    print(reader.dataset[0:10])
    print(reader.test[0:10])

if __name__ == '__main__':
    main()