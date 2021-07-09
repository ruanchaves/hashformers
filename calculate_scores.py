from argparse import ArgumentParser
import json 
from pathlib import Path

class ModelLM(object):

    def __init__(self, model_name_or_path=None, model_type=None, device=None, gpu_batch_size=None):
        if model_type == 'gpt2':
            from gpt2_lm import GPT2LM
            self.model = GPT2LM(model_name_or_path, device=device, gpu_batch_size=gpu_batch_size)
        elif model_type == 'bert':
            from bert_lm import BertLM
            self.model = BertLM(model_name_or_path)

def calculate_scores(
    model_name_or_path,
    model_type,
    source_path
):
    with open(source_path, 'r') as f:
        try:
            source = json.load(f)
        except:
            source = f.read().split('\n')
        
    if all([isinstance(x, str) for x in source]):
        if isinstance(source, dict):
            dataset = list(source.keys())
        elif isinstance(source, list):
            dataset = source
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    model = ModelLM(
        model_name_or_path=model_name_or_path,
        model_type=model_type,
        device='cuda',
        gpu_batch_size=1
    )

    output = {}

    for entry in dataset:
        perplexity = model.get_probs([entry])[0]
        output[entry] = perplexity

    return output

def get_args():
    parser = ArgumentParser()

    parser.add_argument(
        '--model_name_or_path',
        type=str
    )

    parser.add_argument(
        '--model_type',
        type=str
    )

    parser.add_argument(
        '--source',
        type=str
    )

    parser.add_argument(
        '--output',
        type=str
    )

    args = parser.parse_args()

    return args

def main():
    args = get_args()

    scores = calculate_scores(
        args.model_name_or_path,
        args.model_type,
        args.source
    )

    target_path = str(Path(args.output).parent)
    Path(target_path).mkdir(parents=True, exist_ok=True)

    with open(args.output, 'w') as f:
        json.dump(scores, f)

if __name__ == '__main__' :
    main()