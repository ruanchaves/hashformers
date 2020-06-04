from gpt2_encoder import GPT2Encoder
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch 

from configuration_classes import (
   parameters_to_string,
   ModelArguments, 
   DataEvaluationArguments,
   EncoderArguments, 
   CNNArguments,
   MLPArguments
)

from transformers import (
    HfArgumentParser
)

from cnn_model import CNNModel

import logging
import time 

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

class MLP(nn.Module):
    def __init__(self):
        
        super().__init__()

        self.hidden_1 = nn.Linear(4, 64)
        self.hidden_2 = nn.Linear(64, 64)
        self.hidden_3 = nn.Linear(64, 1)
        
    def forward(self, input_features):
        x = F.relu(self.hidden_1(input_features))
        x = F.relu(self.hidden_2(x))
        x = self.hidden_3(x)
        return x

class MLPModel(object):
    def __init__(self, model_args, data_args, cnn_args, mlp_args):
        self.model_args = model_args
        self.data_args = data_args
        self.cnn_args = cnn_args
        self.mlp_args = mlp_args

        self.model = MLP()
        self.model = self.model.to(mlp_args.mlp_device)

        self.cnn_model = CNNModel(model_args, data_args, cnn_args)
        self.cnn_model.from_pretrained()

        self.logger = None

    def _get_features_and_labels(self, gpt2_encoder):

        model_args = self.model_args
        data_args = self.data_args
        mlp_args = self.mlp_args

        gpt2_encoder.compile_generations_df()
        input_ids, _ = gpt2_encoder.get_input_ids_and_labels()

        cnn_probs = [ x.item() for x in self.cnn_model.predict(input_ids) ]
        gpt2_encoder.update_cnn_values(cnn_probs)
        gpt2_encoder.compile_pairs_df()
        features_df = gpt2_encoder.pairs_df[[
            'left_gpt2_score',
            'left_cnn_score',
            'right_gpt2_score',
            'right_cnn_score',
            'distance_score'
        ]]
        features_df_values = features_df.values.tolist()
        features = [ x[0:4] for x in features_df_values ]
        labels = [ x[4] for x in features_df_values ]
        features = torch.tensor(features, dtype=torch.float, device=mlp_args.mlp_device)
        labels = torch.tensor(labels, dtype=torch.float, device=mlp_args.mlp_device)

        return features, labels

    def train(self):
        self.logger.info('MLP training.')
        model_args = self.model_args
        data_args = self.data_args
        mlp_args = self.mlp_args
        cnn_args = self.cnn_args
        model = self.model

        gpt2_encoder = GPT2Encoder(
            model_args.model_name_or_path, 
            data_args.expansions_file, 
            data_args.report_file, 
            data_args.dict_file)
        features, labels = self._get_features_and_labels(gpt2_encoder)

        validation_gpt2_encoder = GPT2Encoder(
            model_args.model_name_or_path, 
            data_args.validation_expansions_file, 
            data_args.validation_report_file, 
            data_args.validation_dict_file)
        validation_features, validation_labels = self._get_features_and_labels(validation_gpt2_encoder)

        num_epochs = mlp_args.mlp_training_epochs
        loss_function = nn.MSELoss()
        loss_function.to(mlp_args.mlp_device)
        optimizer = optim.Adam(model.parameters(), lr=mlp_args.mlp_learning_rate)

        best_validation_loss = float('inf')
        missed_epoch = 0
        for epoch in range(num_epochs):

            start_time = time.time()
            train_loss = 0
            validation_loss = 0

            # Training

            model.train()
            for idx, sequence in enumerate(features):

                model.zero_grad()

                probs = model(sequence)
                target = torch.tensor([labels[idx]], dtype=torch.float, device=mlp_args.mlp_device)

                loss = loss_function(probs, target)
                train_loss += loss.item()

                loss.backward()

                optimizer.step()


            # Validation

            model.eval()
            with torch.no_grad():
                for idx, sequence in enumerate(validation_features):

                    probs = model(sequence)
                    target = torch.tensor([validation_labels[idx]], dtype=torch.float, device=mlp_args.mlp_device)

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
                torch.save(model.state_dict(), mlp_args.mlp_save_path)
            else:
                missed_epoch += 1
                if missed_epoch == mlp_args.mlp_missed_epoch_limit:
                    break

    def predict(self, input_ids):
         with torch.no_grad():
            for sequence in input_ids:
                probs = self.model(sequence)
                yield probs

    def from_pretrained(self, model_path=None):
        if not model_path:
            model_path = self.mlp_args.mlp_save_path
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()    

def main():

    parser = HfArgumentParser((ModelArguments, DataEvaluationArguments, CNNArguments, EncoderArguments, MLPArguments))
    model_args, data_args, cnn_args, encoder_args, mlp_args = parser.parse_args_into_dataclasses()

    # create logger with __file__
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(data_args.logfile)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)  

    logger.info('\n' + parameters_to_string(model_args, data_args, cnn_args, encoder_args, mlp_args))

    mlp_model = MLPModel(model_args, data_args, cnn_args, mlp_args)
    mlp_model.logger = logger
    mlp_model.train()

if __name__ == '__main__':
    main()