import datasets
import os

_CITATION = """\
"""

_DESCRIPTION = """\
"""

_HOMEPAGE = ""

_LICENSE = "[More Information needed]"

_URL = "https://raw.githubusercontent.com/cardiffnlp/xlm-t/main/data/sentiment/english/"

class UMSABConfig(datasets.BuilderConfig):
        def __init__(self, 
            url=None,
            **kwargs):
            if url:
                self.url = url
            else:
                self.url = _URL
            super(UMSABConfig, self).__init__(**kwargs)

class UMSAB(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIG_CLASS = UMSABConfig

    def _info(self):
            
        features = datasets.Features(
            {
                "tweetid": datasets.Value("string"),
                "content": datasets.Value("string"),
                "polarity": datasets.Value("string"),
                "has_hashtag": datasets.Value("bool")
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        my_urls = {
            "train_text": "train_text.txt",
            "train_labels": "train_labels.txt",
            "val_text": "val_text.txt",
            "val_labels": "val_labels.txt",
            "test_text": "test_text.txt",
            "test_labels": "test_labels.txt"
        }

        my_urls = {
            k: os.path.join(self.config.url, v) for k,v in my_urls.items()
        }
        data = dl_manager.download_and_extract(my_urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "textpath": data["train_text"],
                    "labelpath": data["train_labels"],
                    "split": "train"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "textpath": data["val_text"],
                    "labelpath": data["val_labels"],
                    "split": "validation"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "textpath": data["test_text"],
                    "labelpath": data["test_labels"],
                    "split": "test"
                },
            )
        ]

    def _generate_examples(self, textpath, labelpath, split):

        with open(textpath, 'r') as f:
            text = f.read().split('\n')
        with open(labelpath, 'r') as f:
            labels = f.read().split('\n')
        
        assert len(text) == len(labels)

        records = [
            {
                "content": text[idx],
                "polarity": labels[idx]
            } for idx in range(0, len(text))
        ]
        for idx, row in enumerate(records):
            yield idx, {
                "tweetid": str(idx),
                "content": row["content"],
                "polarity": row["polarity"],
                "has_hashtag": "#" in row["content"]
            }