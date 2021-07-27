from transformers import MarianMTModel, MarianTokenizer, AutoTokenizer
from functools import partial
import pandas as pd 
import argparse
import datasets


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default=None
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="Helsinki-NLP/opus-mt-en-es"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000
    )

    args = parser.parse_args()

    return args

def main():
    args = get_args()

    translate_dataset(
        args.dataset_path,
        args.save_path,
        args.model_name,
        batch_size=args.batch_size
    )

def translate_dataset(
    dataset_path,
    save_path,
    model_name="Helsinki-NLP/opus-mt-en-es",
    read_params={
    "header": None,
    "usecols": [0,1,2],
    "names": ["tweetid", "label", "content"],
    "sep": "\t"
},
    content_field="content",
    save_params={
    "sep": "\t",
    "header": None,
    "index": False
},
    device="cuda",
    batch_size=1000
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    model.to(device)

    assert tokenizer != None

    def translate_sentence(sentence, model=None, tokenizer=None):
        if isinstance(sentence, str):
            input = [sentence]
        else:
            input = sentence
        tokens = tokenizer(input, return_tensors="pt", padding=True, truncation=True)
        tokens.to(device)
        translated = model.generate(**tokens)
        translation = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

        if isinstance(sentence, str):
            return translation[0]
        else:
            return translation

    def translate_row(
        row, 
        sentence_field="sentence", 
        model=None, 
        tokenizer=None):
        row[sentence_field] = \
            translate_sentence(row[sentence_field], model=model, tokenizer=tokenizer)
        return row
        
    translate_row_partial = partial(translate_row, model=model, tokenizer=tokenizer)

    translate_row_partial_content_field = \
        partial(translate_row, sentence_field=content_field, model=model, tokenizer=tokenizer)
    
    if dataset_path == "sst":
        dataset = datasets.load_dataset("sst", "default")
        dataset = dataset.map(
            translate_row_partial,
            batched=True,
            batch_size=batch_size)
        dataset.save_to_disk(save_path)
    else:
        df = pd.read_csv(dataset_path, **read_params)
        dataset = datasets.Dataset.from_pandas(df)
        dataset = dataset.map(
            translate_row_partial_content_field,
            batched=True,
            batch_size=batch_size)
        dataset.save_to_disk(save_path)

if __name__ == '__main__':
    main()