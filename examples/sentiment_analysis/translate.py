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

    args = parser.parse_args()

    return args

def main():
    args = get_args()

    translate_from_df(
        args.dataset_path,
        args.save_path,
        args.model_name
    )

def translate_from_df(
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
    device="cuda"
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    model.to(device)

    assert tokenizer != None

    def translate_sentence(sentence, model=None, tokenizer=None):
        tokens = tokenizer([sentence], return_tensors="pt", padding=True, truncation=True)
        tokens.to(device)
        translated = model.generate(**tokens)
        return [tokenizer.decode(t, skip_special_tokens=True) for t in translated][0]

    def translate_row(row, sentence_field="sentence", model=None, tokenizer=None):
        sentence = str(row[sentence_field])
        output = translate_sentence(sentence, model=model, tokenizer=tokenizer)
        row[sentence_field] = str(output)
        return row

    translate_sentence_partial = partial(translate_row, model=model, tokenizer=tokenizer)
    translate_row_partial = partial(translate_sentence, model=model, tokenizer=tokenizer)

    if dataset_path == "sst":
        dataset = datasets.load_dataset("sst", "default")
        dataset = dataset.map(translate_row_partial)
        dataset.save_to_disk(save_path)
    else:
        df = pd.read_csv(dataset_path, **read_params)
        df[content_field] = df[content_field].apply(
            lambda x: translate_sentence_partial(x))
        df.to_csv(save_path, **save_params)

if __name__ == '__main__':
    main()