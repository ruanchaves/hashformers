from typing import get_args
from transformers import MarianMTModel, MarianTokenizer, AutoTokenizer
from functools import partial
import pandas as pd 
import argparse


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
        "--mode",
        type=str,
        default="dataframe"
    )

    args = parser.parse_args()

    return args

def main():
    args = get_args()

    if args.mode == 'dataframe':
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

  def translate_row(row, model=None, tokenizer=None):
    tokens = tokenizer([row], return_tensors="pt", padding=True, truncation=True)
    tokens.to(device)
    translated = model.generate(**tokens)
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated][0]

  df = pd.read_csv(dataset_path, **read_params)
  translate_partial = partial(translate_row, model=model, tokenizer=tokenizer)
  df[content_field] = df[content_field].apply(lambda x: translate_partial(x))
  df.to_csv(save_path, **save_params)

if __name__ == '__main__':
    main()