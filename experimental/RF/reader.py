from utils import window as ngram_generator
from thundersvm import SVC
import joblib
import numpy as np
import functools
import operator
import os
from dotenv import load_dotenv, find_dotenv
from sklearn.ensemble import RandomForestClassifier

class Reader(object):

    def __init__(self, filename):
        self.filename = filename
        with open(filename,'r') as f:
            self.lines = f.read().split('\n')
        self.lines = [ x.strip() for x in self.lines ]
        self.features = []
        self.NULL_CHARACTER = 2
        self.PADDING_CHARACTER = 1

    def build_feature_set(self):
        for item in self.lines:
            words, spaces = self.string_to_number(item)
            for feature in self.generate_features(words, spaces):
                self.features.append(feature)

    def get_features(self, tokens):
        words, spaces = self.string_to_number(tokens)
        features = []
        for feature in self.generate_features(words, spaces):
            features.append(feature)
        output = [ x[0:-1] for x in features ]
        return output
        
    def string_to_number(self, line):
        chars = [ x for x in line ]
        new_chars = []
        for idx, item in enumerate(chars):
            if item.isspace():
                continue
            else:
                new_chars.append(ord(item))
            try:
                if chars[idx+1].isspace():
                    new_chars.append(0)
                else:
                    new_chars.append(1)
            except Exception as e:
                pass
        words = new_chars[0:][::2]
        spaces = new_chars[1:][::2]
        return words, spaces

    def generate_features(self, words, spaces, step=4):
        assert(step % 2 == 0)
        padding_length = ( step // 2 ) + 1
        padding = [self.PADDING_CHARACTER] * padding_length
        words = padding + words + padding
        for idx in range(0, len(words)):
            try:
                current_vector = words[idx:idx+step]
                bigrams = [ x for x in ngram_generator(current_vector, 2) ]
                unigrams = [ 
                    [ current_vector[len(current_vector)//2], self.NULL_CHARACTER ],
                    [ self.NULL_CHARACTER, current_vector[(len(current_vector)//2)+1]]
                ]
                feature = bigrams + unigrams 
                feature = self.szudzik_function(feature) 
                feature = feature + [spaces[idx]]
                yield feature
            except IndexError:
                break

    def szudzik_function(self, list_of_lists):
        output = []
        for item in list_of_lists:
            a = item[0]
            b = item[1]
            if a >= b:
                result = a * a + a + b
            else:
                result = a + b * b
            output.append(result)
        return output

def train_model(training_set, save_path):
    reader = Reader(training_set)
    reader.build_feature_set()
    X = [ x[0:-1] for x in reader.features ]
    Y = [ x[-1] for x in reader.features ]
    clf = SVC()
    clf.fit(X, Y)
    joblib.dump(clf, save_path) 

if __name__ == '__main__':
    load_dotenv(find_dotenv())
    train_model(os.environ['TRAINING_SET'], os.environ['SAVE_PATH'])
