local base_url = "https://raw.githubusercontent.com/ruanchaves/xlm-t/main/data/sentiment";
local batch_size = 1;

local settings = [
    {
        language: "arabic",
        translation_model: "Helsinki-NLP/opus-mt-ar-en",
        segmenter: "aubmindlab/aragpt2-large",
        reranker: "aubmindlab/bert-large-arabertv2",
        spacy_model: "xx_sent_ud_sm"
    },
    {
        language: "french",
        translation_model: "Helsinki-NLP/opus-mt-fr-en",
        segmenter: "antoiloui/belgpt2",
        reranker: "camembert/camembert-large",
        spacy_model: "fr_dep_news_trf"
    },
    {
        language: "german",
        translation_model: "Helsinki-NLP/opus-mt-de-en",
        segmenter: "dbmdz/german-gpt2",
        reranker: "bert-base-german-cased",
        spacy_model: "de_dep_news_trf"
    },
    {
        language: "hindi",
        translation_model: "Helsinki-NLP/opus-mt-hi-en",
        segmenter: "surajp/gpt2-hindi",
        reranker: "ai4bharat/indic-bert",
        spacy_model: "xx_sent_ud_sm" 
    },
    {
        language: "italian",
        translation_model: "Helsinki-NLP/opus-mt-it-en",
        segmenter: "GroNLP/gpt2-small-italian",
        reranker: "dbmdz/bert-base-italian-xxl-cased",
        spacy_model: "it_core_news_lg"    
    },
    {
        language: "portuguese",
        translation_model: "Helsinki-NLP/opus-mt-roa-en",
        segmenter: "pierreguillou/gpt2-small-portuguese",
        reranker: "neuralmind/bert-large-portuguese-cased",
        spacy_model: "pt_core_news_lg"    
    },
    {
        language: "spanish",
        translation_model: "Helsinki-NLP/opus-mt-es-en",
        segmenter: "mrm8488/spanish-gpt2",
        reranker: "dccuchile/bert-base-spanish-wwm-cased",
        spacy_model: "es_dep_news_trf"
    }
];

local Params(
    base_url,
    language,
    translation_model,
    batch_size,
    segmenter,
    reranker,
    spacy_model
) = {
    translation_model_name_or_path: translation_model,
    translation_model_batch_size: batch_size,
    translation_model_device: "cuda",
    log_level: "INFO",
    dataset_reader: "umsab.py",
    dataset_url: "%s/%s" % [base_url, language],
    decoder_model_name_or_path: segmenter,
    encoder_model_name_or_path: reranker,
    spacy_model: spacy_model,
    dataset_save_path: "./sample_%s" % language,
    sample: 100
};

local config = [
    Params(
        base_url,
        x["language"],
        x["translation_model"],
        batch_size,
        x["segmenter"],
        x["reranker"],
        x["spacy_model"]
    ) for x in settings
];


{
    [std.format("%d.json", i)]: config[i] for i in std.range(0, std.length(config)-1)
}