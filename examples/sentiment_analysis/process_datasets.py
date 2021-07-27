import datasets

def process_sst_dataset(dataset, split="test"):
    sst_df = dataset[split].to_pandas()[["sentence", "label"]]
    sst_df = sst_df.rename(
        columns={
            "sentence": "sentence1"
        }
    )
    def round_labels(x):
        return int((round(x * 2) / 2) * 2)
    
    sst_df['label'] = sst_df['label'].apply(round_labels)

    return sst_df 

def process_semeval_dataset(dataset):
    semeval_df = dataset.to_pandas()[["content", "label"]]
    semeval_df = semeval_df.rename(
        columns={
            "content": "sentence1"
        }
    )

    replacement_dict = {
        "negative": 0,
        "neutral": 1,
        "positive": 2,
    }
    semeval_df["label"] = semeval_df["label"].apply(
        lambda x:
            replacement_dict[x]
    )

    return semeval_df

def process_tass_dataset(dataset, split='test'):
    tass_df = dataset[split].to_pandas()[["content", "polarity"]]
    tass_df = tass_df.rename(
        columns={
            "content": "sentence1",
            "polarity": "label"
        }
    )
    replacement_dict = {
        "0": 0,
        "2": 1,
        "1": 2,
    }
    tass_df["label"] = tass_df["label"].apply(
        lambda x:
            replacement_dict[x]
    )

    return tass_df

def main():

    semeval_dev = datasets.load_from_disk("semeval2013_es_dev")
    semeval_train = datasets.load_from_disk("semeval2013_es_train")
    tass = datasets.load_from_disk("tass")
    sst = datasets.load_from_disk("sst_es")

    params = {
        "index": False
    }

    semeval_dev_df = process_sst_dataset(semeval_dev)
    semeval_dev_df.to_csv("semeval_dev.csv", **params)

    semeval_train_df = process_semeval_dataset(semeval_train)
    semeval_train_df.to_csv("semeval_train.csv", **params)

    tass_test_df = process_tass_dataset(tass)
    tass_test_df.to_csv("tass_test.csv", **params)

    sst_train_df = process_sst_dataset(sst, split="train")
    sst_train_df.to_csv("sst_train.csv", **params)

    sst_dev_df = process_sst_dataset(sst, split="dev")
    sst_dev_df.to_csv("sst_dev.csv", **params)

    sst_test_df = process_sst_dataset(sst, split="test")
    sst_test_df.to_csv("sst_test.csv", **params)

if __name__ == '__main__':
    main()