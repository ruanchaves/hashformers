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
    parameters_to_string,
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

import logging 

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


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
            cnn_args.cnn_device)

        self.model = self.model.to(cnn_args.cnn_device)

        self.logger = None

    def train(self):
        self.logger.info('CNN training.')
        
        model_args = self.model_args
        data_args = self.data_args
        cnn_args = self.cnn_args
        model = self.model

        gpt2_encoder = GPT2Encoder(
            model_args.model_name_or_path, 
            data_args.expansions_file, 
            data_args.report_file, 
            data_args.dict_file)

        gpt2_encoder.compile_generations_df()
        input_ids, labels = gpt2_encoder.get_input_ids_and_labels()        
    
        validation_gpt2_encoder = GPT2Encoder(
            model_args.model_name_or_path, 
            data_args.validation_expansions_file, 
            data_args.validation_report_file, 
            data_args.validation_dict_file)

        validation_gpt2_encoder.compile_generations_df()
        validation_input_ids, validation_labels = validation_gpt2_encoder.get_input_ids_and_labels()   

        self.logger.info(f'The model has {count_parameters(model):,} trainable parameters')

        num_epochs = cnn_args.cnn_training_epochs
        loss_function = nn.CrossEntropyLoss()
        loss_function.to(cnn_args.cnn_device)
        optimizer = optim.Adam(model.parameters(), lr=cnn_args.cnn_learning_rate)

        best_validation_loss = float('inf')
        missed_epoch = 0
        for epoch in range(num_epochs):

            start_time = time.time()
            train_loss = 0
            validation_loss = 0

            # Training

            model.train()
            for idx, sequence in enumerate(input_ids):

                model.zero_grad()

                probs = model(sequence)
                target = torch.tensor([labels[idx]], dtype=torch.long, device=cnn_args.cnn_device)

                loss = loss_function(probs, target)
                train_loss += loss.item()

                loss.backward()

                optimizer.step()


            # Validation

            model.eval()
            with torch.no_grad():
                for idx, sequence in enumerate(validation_input_ids):

                    probs = model(sequence)
                    target = torch.tensor([validation_labels[idx]], dtype=torch.long, device=cnn_args.cnn_device)

                    loss = loss_function(probs, target)
                    validation_loss += loss.item()

            # Saving

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            logging_message = "\nEpoch {0} : {1}m{2}s \n Train loss: {3} \n Validation loss: {4}"\
                .format(epoch, epoch_mins, epoch_secs, train_loss, validation_loss)
            self.logger.info(logging_message)


            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                torch.save(model.state_dict(), cnn_args.cnn_save_path)
            else:
                missed_epoch += 1
                if missed_epoch == cnn_args.cnn_missed_epoch_limit:
                    break

    def predict(self, input_ids):
        with torch.no_grad():
            for sequence in input_ids:
                probs = self.model(sequence)
                yield probs

    def from_pretrained(self, model_path=None):
        if not model_path:
            model_path = self.cnn_args.cnn_save_path
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

def main():

    parser = HfArgumentParser((ModelArguments, DataEvaluationArguments, CNNArguments))
    model_args, data_args, cnn_args = parser.parse_args_into_dataclasses()
   
    # create logger with __file__
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(data_args.logfile)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)   

    logger.info('\n' + parameters_to_string(model_args, data_args, cnn_args))

    cnn_model = CNNModel(model_args, data_args, cnn_args)
    cnn_model.logger = logger
    cnn_model.train()

if __name__ == '__main__':
    main()

