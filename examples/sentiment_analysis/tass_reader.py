import datasets
import xml.etree.ElementTree as ET

_CITATION = """\
"""

_DESCRIPTION = """\
"""

_HOMEPAGE = ""

_LICENSE = "[More Information needed]"

class TassConfig(datasets.BuilderConfig):
        def __init__(self, url=None, **kwargs):
            self.url = url
            super(TassConfig, self).__init__(**kwargs)

class Tass(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIG_CLASS = TassConfig

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
        data = dl_manager.download(my_urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data['test'],
                    "split": "test"
                },
            )
        ]

    def _generate_examples(self, filepath, split):
        polarity_dict = {
            'P': "1",
            'N': "0",
            'NEU': "2"
        }
        tree = ET.parse(filepath)
        root = tree.getroot()
        for tweet in root.iter('tweet'):

            sentiments = tweet.find('sentiments')
            polarity = sentiments[0].find('value').text
            polarity = polarity_dict.get(polarity, None)
            if not polarity:
                continue

            content = tweet.find('content').text
            tweet_id = tweet.find('tweetid').text
            
            yield tweet_id, {
                "tweetid": tweet_id,
                "content": content,
                "polarity": polarity,
                "has_hashtag": "#" in content
            }