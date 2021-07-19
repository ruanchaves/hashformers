import datasets
import xml.etree.ElementTree as ET
import os
import pandas as pd

_CITATION = """\
"""

_DESCRIPTION = """\
"""

_HOMEPAGE = ""

_LICENSE = "[More Information needed]"

_URL = "http://alt.qcri.org/semeval2017/task4/data/uploads/semeval2017-task4-test.zip"

class SemEvalConfig(datasets.BuilderConfig):
        def __init__(self, 
            url=None, 
            skip_neutral=True, 
            positive_label="POSITIVE",
            negative_label="NEGATIVE",
            neutral_label="NEUTRAL",
            **kwargs):
            if url:
                self.url = url
            else:
                self.url = _URL
            self.skip_neutral = skip_neutral
            self.positive_label = str(positive_label)
            self.negative_label = str(negative_label)
            self.neutral_label = str(neutral_label)
            super(SemEvalConfig, self).__init__(**kwargs)

class SemEval2017(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIG_CLASS = SemEvalConfig

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
            "test": self.config.url
        }
        data = dl_manager.download_and_extract(my_urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(
                        data['test'],
                        "SemEval2017-task4-test/SemEval2017-task4-test.subtask-A.english.txt"
                    ),
                    "split": "test"
                },
            )
        ]

    def _generate_examples(self, filepath, split):
        polarity_dict = {
            'positive': self.config.positive_label,
            'neutral': self.config.neutral_label,
            'negative': self.config.negative_label
        }
        df = pd.read_csv(
            filepath,
            sep='\t',
            header=None,
            usecols=[0,1,2],
            names=["tweetid", "original_polarity", "content"])
        
        df["polarity"] = df["original_polarity"].apply(
            lambda x: polarity_dict[x]
        )

        df = df.astype(str)

        records = df.to_dict('records')
        for idx, row in enumerate(records):
            if self.config.skip_neutral \
                and row["polarity"] == self.config.neutral_label:
                continue
            yield row["tweetid"], {
                "tweetid": row["tweetid"],
                "content": row["content"],
                "polarity": row["polarity"],
                "has_hashtag": "#" in row["content"]
            }