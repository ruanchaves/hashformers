local base_url = "https://raw.githubusercontent.com/ruanchaves/xlm-t/main/data/sentiment"

local settings = [
    {
        language: "arabic",
        translation_model: "Helsinki-NLP/opus-mt-ar-en",
        segmenter: "aubmindlab/aragpt2-mega",
        reranker: "aubmindlab/bert-large-arabertv2"
    },
    {
        language: "french",
        translation_model: "Helsinki-NLP/opus-mt-fr-en",
        segmenter: "antoiloui/belgpt2",
        reranker: "camembert/camembert-large"
    },
    {
        language: "german",
        translation_model: "Helsinki-NLP/opus-mt-de-en",
        segmenter: "dbmdz/german-gpt2",
        reranker: "bert-base-german-cased"
    },
    {
        language: "hindi",
        translation_model: "Helsinki-NLP/opus-mt-hi-en",
        segmenter: "surajp/gpt2-hindi",
        reranker: "ai4bharat/indic-bert"    
    },
    {
        language: "italian",
        translation_model: "Helsinki-NLP/opus-mt-it-en",
        segmenter: "GroNLP/gpt2-small-italian",
        reranker: "dbmdz/bert-base-italian-xxl-cased"    
    },
    {
        language: "portuguese",
        translation_model: "Helsinki-NLP/opus-mt-roa-en",
        segmenter: "pierreguillou/gpt2-small-portuguese",
        reranker: "neuralmind/bert-large-portuguese-cased"    
    },
    {
        language: "spanish",
        translation_model: "Helsinki-NLP/opus-mt-es-en",
        segmenter: "mrm8488/spanish-gpt2",
        reranker: "dccuchile/bert-base-spanish-wwm-cased"
    }
]