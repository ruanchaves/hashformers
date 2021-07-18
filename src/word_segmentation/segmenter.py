import itertools
from word_segmentation.beamsearch.algorithm import Beamsearch
from word_segmentation.beamsearch.reranker import Reranker
from word_segmentation.beamsearch.data_structures import enforce_prob_dict
from word_segmentation.ensemble.top2_fusion import top2_ensemble
from spacy.tokenizer import _get_regex_pattern
import spacy
import re 
import itertools
from spacy.matcher import Matcher

class WordSegmenter(object):

    def __init__(
        self,
        decoder_model_name_or_path = "gpt2",
        decoder_model_type = "gpt2",
        decoder_device = "cuda",
        decoder_gpu_batch_size = 1,
        encoder_model_name_or_path = "bert-base-uncased",
        encoder_model_type = "bert",
        spacy_language = "en"
    ):
        if spacy_language:
            self.nlp = spacy.load(spacy_language, disable=['parser', 'tagger', 'ner'])
            re_token_match = _get_regex_pattern(self.nlp.Defaults.token_match)
            self.nlp.tokenizer.token_match = re.compile(f"({re_token_match}|#\w+|\w+-\w+)").match

        self.decoder_model = Beamsearch(
        model_name_or_path=decoder_model_name_or_path,
        model_type=decoder_model_type,
        device=decoder_device,
        gpu_batch_size=decoder_gpu_batch_size
    )

        if encoder_model_name_or_path:
            self.encoder_model = Reranker(
                model_name_or_path=encoder_model_name_or_path,
                model_type=encoder_model_type
            )
        else:
            self.encoder_model = None

    def replace_word(self, text, word, replacement):
        doc = self.nlp(text)
        matches = [
            (token.idx, len(word)) for token in doc 
                if token.text[1:].lower()==word 
                and token.text.startswith("#")
        ]
        matches = sorted(matches, key=lambda x:-x[0])
        for i, l in matches: 
            text = text[:i] + replacement + text[i+l:]
        return text

    def process_hashtags(
        self,
        text_list,
        topk=20,
        steps=13,
        alpha=0.222,
        beta=0.111,
        use_encoder=True
    ):

        hashtag_dict = {}

        def filter_hashtags(tokens):
            return [ 
                x.text[1:] for x in tokens 
                    if x.text.startswith("#") 
            ]
        
        def replace_hashtags(tokens):
            return [ 
                hashtag_dict.get(x.text[1:], x.text) 
                    if x.text.startswith("#") else x.text for x in tokens
            ]

        hashtags = [ filter_hashtags(self.nlp(x)) for x in text_list ]
        hashtag_list = list(itertools.chain.from_iterable(hashtags)) #flatten
        segmentations = self.segment(
            hashtag_list,
            topk=topk,
            steps=steps,
            alpha=alpha,
            beta=beta,
            use_encoder=use_encoder)

        for idx, item in enumerate(segmentations):
            hashtag_dict.update({
                hashtags[idx] : segmentations[idx]
            })

        output = [ replace_hashtags(self.nlp(x)) for x in text_list ]
        return output

    def segment(
            self,
            word_list,
            topk=20,
            steps=13,
            alpha=0.222,
            beta=0.111,
            use_encoder=True):

        decoder_run = self.decoder_model.run(
            word_list,
            topk=topk,
            steps=steps
        )
        
        if self.encoder_model:
            encoder_run = self.encoder_model.rerank(decoder_run)

            ensemble = top2_ensemble(
                decoder_run,
                encoder_run,
                alpha=alpha,
                beta=beta
            )
        
        if self.encoder_model and use_encoder:
            ensemble_prob_dict = enforce_prob_dict(
                ensemble,
                score_field="ensemble_rank")
            segs = ensemble_prob_dict.get_segmentations(
                astype="list",
                gold_array=word_list
            )
        else:
            decoder_prob_dict = enforce_prob_dict(
                decoder_run,
                score_field="score"
            )
            segs = decoder_prob_dict.get_segmentation(
                astype="list",
                gold_array=word_list
            )

        return segs