from gpt2_lm import GPT2LM
import pandas as pd
class Ranking(object):

    def __init__(self, model_name_or_path):
        self.gpt2 = GPT2LM('gpt2')

    def calculate_score(self, a, b):
        gpt2_input = a + b
        score = self.gpt2.get_probs([gpt2_input])
        return score[0]

df = pd.read_csv('entropy.csv')
print(df.head())
rank = Ranking('gpt2')

df['score'] = df['g1'].combine(df['g2'], rank.calculate_score)

df.to_csv('entropy2.csv')