from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from hashformers.experiments.evaluation import evaluate_df
import pandas as pd

def main():
    text_processor = TextPreProcessor(
        # terms that will be normalized
        normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
            'time', 'url', 'date', 'number'],
        # terms that will be annotated
        annotate={"hashtag", "allcaps", "elongated", "repeated",
            'emphasis', 'censored'},
        fix_html=True,  # fix HTML tokens
        
        # corpus from which the word statistics are going to be used 
        # for word segmentation 
        segmenter="twitter", 
        
        # corpus from which the word statistics are going to be used 
        # for spell correction
        corrector="twitter", 
        
        unpack_hashtags=True,  # perform word segmentation on hashtags
        unpack_contractions=True,  # Unpack contractions (can't -> can not)
        spell_correct_elong=False,  # spell correction for elongated words
        
        # select a tokenizer. You can use SocialTokenizer, or pass your own
        # the tokenizer, should take as input a string and return a list of tokens
        tokenizer=SocialTokenizer(lowercase=True).tokenize,
        
        # list of dictionaries, for replacing tokens extracted from the text,
        # with other expressions. You can pass more than one dictionaries.
        dicts=[emoticons]
    )

    dataset_dict = {
        "BOUN": "https://raw.githubusercontent.com/prashantkodali/HashSet/master/datasets/boun-celebi-et-al.csv",
        "STAN-Dev": "https://raw.githubusercontent.com/prashantkodali/HashSet/master/datasets/stan-dev-celebi-etal.csv"
    }

    for key, value in dataset_dict.items():
        df = pd.read_csv(value, header=None, names=["characters", "gold"])
        df["segmentation"] = df["characters"].apply(lambda x: " ".join(text_processor.pre_process_doc("#" + x)[1:-1]))
        result = evaluate_df(
            df,
            gold_field="gold",
            segmentation_field="segmentation"
        )
        print(key, result)

    dataset_dict = {
        "Distant-sampled": "https://raw.githubusercontent.com/prashantkodali/HashSet/master/datasets/hashset/HashSet-Distant-sampled.csv"
    }


    for key, value in dataset_dict.items():
        df = pd.read_csv(value, sep=',')
        df_1 = df[["Unsegmented_hashtag", "Segmented_hashtag"]]
        df_1 = df_1.rename(columns={"Unsegmented_hashtag": "characters", "Segmented_hashtag": "gold"})
        df_2 = df[["Unsegmented_hashtag_lowerCase","Segmented_hashtag_lowerCase"]]
        df_2 = df_2.rename(columns={"Unsegmented_hashtag_lowerCase": "characters","Segmented_hashtag_lowerCase": "gold"})
        df_1["segmentation"] = df_1["characters"].apply(lambda x: " ".join(text_processor.pre_process_doc("#" + x)[1:-1]))
        df_2["segmentation"] = df_2["characters"].apply(lambda x: " ".join(text_processor.pre_process_doc("#" + x)[1:-1]))
        result_1 = evaluate_df(
            df_1,
            gold_field="gold",
            segmentation_field="segmentation"
        )
        result_2 = evaluate_df(
            df_2,
            gold_field="gold",
            segmentation_field="segmentation"
        )
        print("HashSet sample", result_1)
        print("Hashset sample, lowercase", result_2)

    dataset_dict = {
        "Manual": "https://raw.githubusercontent.com/prashantkodali/HashSet/master/datasets/hashset/HashSet-Manual.csv"
    }

    for key, value in dataset_dict.items():
        df = pd.read_csv(value, sep=',')
        df = df[["Hashtag", "Final Segmentation"]]
        df = df.rename(columns={"Hashtag": "characters", "Final Segmentation": "gold"})
        df["segmentation"] = df["characters"].apply(lambda x: " ".join(text_processor.pre_process_doc("#" + x)[1:-1]))
        result = evaluate_df(
            df,
            gold_field="gold",
            segmentation_field="segmentation"
        )
        print("HashSet Manual", result)

if __name__ == '__main__':
    main()