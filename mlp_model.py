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
import logging.config
import time 

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

class MLP(nn.Module):
    def __init__(self):
        
        super().__init__()

        self.hidden_1 = nn.Linear(14, 15)
        self.hidden_2 = nn.Linear(15, 7)
        self.hidden_3 = nn.Linear(7, 1)
        
    def forward(self, input_features):
        input_features = input_features.unsqueeze(1)
        x = F.relu(self.hidden_1(input_features))
        x = F.relu(self.hidden_2(x))
        x = self.hidden_3(x)
        return x

class Batch(object):

    def __init__(self, input_ids, label, device='cuda', dtype=torch.float):
        self.text = input_ids
        self.label = torch.tensor(label, dtype=torch.float, device=device)


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

        self.logger.debug("Compiling generations df.")
        gpt2_encoder.compile_generations_df()
        input_ids, _ = gpt2_encoder.get_input_ids_and_labels(trim=False, pad=True)

        batch_size = 1
        input_ids = [torch.stack(input_ids[i:i+batch_size]) for i in range(0, len(input_ids), batch_size)]

        cnn_probs = [ x.item() for x in self.cnn_model.predict(input_ids) ]
        gpt2_encoder.update_cnn_values(cnn_probs)
    
        self.logger.debug("Compiling pairs df.")
        pairs_array = gpt2_encoder.compile_pairs_df()
        features = [ x[0:-1] for x in pairs_array ]
        labels = [ x[-1] for x in pairs_array ]

        self.logger.debug("Converting features and labels.")
        features = torch.tensor(features, dtype=torch.float, device=mlp_args.mlp_device)
        labels = torch.tensor(labels, dtype=torch.float, device=mlp_args.mlp_device)

        return features, labels

    def train(self):
        model_args = self.model_args
        data_args = self.data_args
        mlp_args = self.mlp_args
        cnn_args = self.cnn_args
        model = self.model

        self.logger.info('Initializing training set encoder.')
        gpt2_encoder = GPT2Encoder(
            model_args.model_name_or_path, 
            data_args.expansions_file, 
            data_args.report_file, 
            data_args.dict_file)

        self.logger.info('Extracting features from training set.')
        features, labels = self._get_features_and_labels(gpt2_encoder)

        batch_size = mlp_args.mlp_batch_size
    
        features = torch.split(features, batch_size)
        labels = torch.split(labels, batch_size)

        features = [x for x in features if len(x) == batch_size]
        labels = [x for x in labels if len(x) == batch_size]    
        
        features = torch.stack(features)
        labels = torch.stack(labels)    

        self.logger.info('Processing validation set.')
        validation_gpt2_encoder = GPT2Encoder(
            model_args.model_name_or_path, 
            data_args.validation_expansions_file, 
            data_args.validation_report_file, 
            data_args.validation_dict_file)
        validation_features, validation_labels = self._get_features_and_labels(validation_gpt2_encoder) 

        validation_features = torch.split(validation_features, batch_size)
        validation_labels = torch.split(validation_labels, batch_size)

        validation_features = [x for x in validation_features if len(x) == batch_size ]
        validation_labels = [x for x in validation_labels if len(x) == batch_size ]

        validation_features = torch.stack(validation_features)
        validation_labels = torch.stack(validation_labels)

        training_input = [ Batch(x,y) for x,y in list(zip(features, labels)) ]    
        validation_input = [ Batch(x,y) for x,y in list(zip(validation_features, validation_labels)) ]    


        num_epochs = mlp_args.mlp_training_epochs
        loss_function = nn.MSELoss()
        loss_function.to(mlp_args.mlp_device)
        optimizer = optim.Adam(model.parameters(), lr=mlp_args.mlp_learning_rate)

        self.logger.info('MLP training.')
        best_validation_loss = float('inf')
        missed_epoch = 0
        for epoch in range(num_epochs):
            self.logger.info("Starting epoch {0}.".format(epoch))
            start_time = time.time()
            train_loss = 0
            validation_loss = 0

            # Training

            train_loss, train_acc = self._train_epoch(model, training_input, optimizer, loss_function)

            # Validation

            validation_loss, validation_acc = self._evaluate_epoch(model, validation_input, loss_function)

            # Saving

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            self.logger.info(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            self.logger.info(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            self.logger.info(f'\t Val. Loss: {validation_loss:.3f} |  Val. Acc: {validation_acc*100:.2f}%')

            if validation_loss < best_validation_loss:
                missed_epoch = 0
                best_validation_loss = validation_loss
                self.logger.info("Saving model to {0}".format(mlp_args.mlp_save_path))
                torch.save(model.state_dict(), mlp_args.mlp_save_path)
            else:
                missed_epoch += 1
                if missed_epoch == mlp_args.mlp_missed_epoch_limit:
                    self.logger.info("Reached missed epoch limit.")
                    torch.save(model.state_dict(), mlp_args.mlp_save_path)
                    break
        else:
            self.logger.info("Saving model to {0}".format(mlp_args.mlp_save_path))
            torch.save(model.state_dict(), mlp_args.mlp_save_path)           

    def _train_epoch(self, model, iterator, optimizer, criterion):     
        epoch_loss = 0
        epoch_acc = 0
        
        model.train()
        
        for batch in iterator:
            
            optimizer.zero_grad()
            
            predictions = model(batch.text)

            try:
                assert(predictions.shape == batch.label.shape)
            except:
                predictions = predictions.flatten()
                batch.label = batch.label.flatten()
                assert(predictions.shape == batch.label.shape)


            loss = criterion(predictions, batch.label.flatten())
            acc = self.binary_accuracy(predictions, batch.label)
            
            loss.backward()
            
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def _evaluate_epoch(self, model, iterator, criterion):
        
        epoch_loss = 0
        epoch_acc = 0
        
        model.eval()
        
        with torch.no_grad():
        
            for batch in iterator:

                predictions = model(batch.text).flatten()

                try:
                    assert(predictions.shape == batch.label.shape)
                except:
                    predictions = predictions.flatten()
                    batch.label = batch.label.flatten()
                    assert(predictions.shape == batch.label.shape)


                loss = criterion(predictions, batch.label.flatten())
                
                acc = self.binary_accuracy(predictions, batch.label)

                epoch_loss += loss.item()
                epoch_acc += acc.item()
            
        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def binary_accuracy(self, preds, y):
        """
        Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
        """

        #round predictions to the closest integer
        rounded_preds = torch.round(torch.sigmoid(preds))
        correct = (rounded_preds == y).float() #convert into float for division 
        acc = correct.sum() / len(correct)
        return acc


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

    # logging.config.fileConfig('logging.conf', \
    #     defaults={'logfilename': data_args.logfile})
    logger = logging.getLogger(__file__)

    logger.info('\n' + parameters_to_string(model_args, data_args, cnn_args, encoder_args, mlp_args))

    mlp_model = MLPModel(model_args, data_args, cnn_args, mlp_args)
    mlp_model.logger = logger
    mlp_model.train()

if __name__ == '__main__':
    main()