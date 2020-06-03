import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import time
import itertools
import torch.optim as optim
from types import SimpleNamespace
from collections import defaultdict
import pandas as pd

from configuration_classes import (
    ModelArguments,
    DataEvaluationArguments,
    CNNArguments
)

from transformers import (
    GPT2Model,
    GPT2Tokenizer,
    HfArgumentParser
)

from gpt2_encoder import GPT2Encoder

def parse_to_integer_list(comma_separated_values):
    return [int(x) for x in comma_separated_values.split(',') ]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class CNN(nn.Module):
    def __init__(self, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, model_name_or_path, device):
        
        super().__init__()
        self.device = device
        self.gpt2 = GPT2Model.from_pretrained(model_name_or_path)
        self.gpt2.to(device)

        self.conv_0 = nn.Conv2d(in_channels = 1, 
                                out_channels = n_filters, 
                                kernel_size = (filter_sizes[0], embedding_dim))
        self.conv_1 = nn.Conv2d(in_channels = 1, 
                                out_channels = n_filters, 
                                kernel_size = (filter_sizes[1], embedding_dim))
        self.conv_2 = nn.Conv2d(in_channels = 1, 
                                out_channels = n_filters, 
                                kernel_size = (filter_sizes[2], embedding_dim))
        
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
    def forward(self, input_ids):
        embedded = self._get_embedding(input_ids)
        embedded = embedded.unsqueeze(0)
        embedded = embedded.unsqueeze(0)
        
        conved_0 = F.relu(self.conv_0(embedded).squeeze(3))
        conved_1 = F.relu(self.conv_1(embedded).squeeze(3))
        conved_2 = F.relu(self.conv_2(embedded).squeeze(3))
            
        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        
        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim = 1))
            
        return self.fc(cat)

    def _get_embedding(self, input_ids):
        input_ids = input_ids.to(self.device)
        gpt_output = self.gpt2(input_ids)[0]
        gpt_output = gpt_output.view(gpt_output.shape[0],-1)
        return gpt_output

class CNNModel(object):
    def __init__(self, model_args, data_args, cnn_args):

        self.model_args = model_args
        self.data_args = data_args
        self.cnn_args = cnn_args

        self.model = CNN(cnn_args.token_embedding_size, 
            cnn_args.n_filters, 
            parse_to_integer_list(cnn_args.filter_sizes), 
            cnn_args.output_dim,
            cnn_args.dropout,
            model_args.model_name_or_path,
            cnn_args.device)

        self.model = self.model.to(cnn_args.device)

        self.gpt2 = GPT2Encoder(model_args.model_name_or_path, data_args.expansions_file, data_args.report_file, data_args.dict_file)
        self.input_ids, self.labels = self.gpt2.get_input_ids_and_labels()

    def train(self):

        model_args = self.model_args
        data_args = self.data_args
        cnn_args = self.cnn_args

        model = self.model
        input_ids = self.input_ids
        labels = self.labels

        print(f'The model has {count_parameters(model):,} trainable parameters')

        num_epochs = cnn_args.cnn_training_epochs
        loss_function = nn.CrossEntropyLoss()
        loss_function.to(cnn_args.device)
        optimizer = optim.Adam(model.parameters(), lr=model_args.learning_rate)

        model.train()
        for epoch in range(num_epochs):
            print("Epoch" + str(epoch + 1))
            train_loss = 0
            for idx, sequence in enumerate(input_ids):

                model.zero_grad()

                probs = model(sequence)
                target = torch.tensor([labels[idx]], dtype=torch.long, device=cnn_args.device)

                loss = loss_function(probs, target)
                train_loss += loss.item()

                loss.backward()

                optimizer.step()

        torch.save(model.state_dict(), cnn_args.cnn_save_path)

    def from_pretrained(self, model_path=None):
        if not model_path:
            model_path = self.cnn_args.cnn_save_path
        self.model.load_state_dict(model_path)
        self.model.eval()

    def predict(self, input_ids):
        with torch.no_grad():
            for sequence in input_ids:
                probs = self.model(sequence)
                print(probs)

def predict():
    parser = HfArgumentParser((ModelArguments, DataEvaluationArguments, CNNArguments))
    model_args, data_args, cnn_args = parser.parse_args_into_dataclasses()

    gpt2 = GPT2Encoder(model_args.model_name_or_path, data_args.expansions_file, data_args.report_file, data_args.dict_file)
    input_ids, _ = gpt2.get_input_ids_and_labels()
    
    model = CNN(cnn_args.token_embedding_size, 
        cnn_args.n_filters, 
        parse_to_integer_list(cnn_args.filter_sizes), 
        cnn_args.output_dim,
        cnn_args.dropout,
        model_args.model_name_or_path,
        cnn_args.device)    

    model.load_state_dict(cnn_args.cnn_save_path)
    model.eval()
    with torch.no_grad():
        for idx, sequence in enumerate(input_ids):
            probs = model(sequence)
            print(probs)

def main():
    parser = HfArgumentParser((ModelArguments, DataEvaluationArguments, CNNArguments))
    model_args, data_args, cnn_args = parser.parse_args_into_dataclasses()
    cnn_model = CNNModel(model_args, data_args, cnn_args)
    if cnn_args.do_training:
        cnn_model.train()
    if cnn_args.do_evaluation:
        cnn_model.predict()

if __name__ == '__main__':
    main()

