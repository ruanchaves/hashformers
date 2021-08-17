local BASE_PATH = std.extVar("BASE_PATH");
local LANGUAGE = std.extVar("LANGUAGE");

local Params(
    base_path,
    language,
    model,
    content_field,
    segmented_content_field
) = {
    dataset_load_path: "%s/translation/%s" % [base_path, language],
    do_eval: true,
    run_classifier: true,
    run_segmenter: false,
    sentiment_model: model,
    content_field: content_field,
    segmented_content_field: segmented_content_field,
    metrics: "%s/sentiment_analysis/sentiment_metrics.py" % base_path 
};

local Experiments(base_path, language, model) = [
    Params(
        base_path,
        language,
        model,
        "translation",
        "translation_replaced_hashtags"
    ),
    Params(
        base_path,
        language,
        model,
        "translation",
        "translation_segmented_hashtags"
    )
];

local models = [
        "cardiffnlp/twitter-roberta-base-sentiment",
        "distilbert-base-uncased-finetuned-sst-2-english",
        "textattack/roberta-base-SST-2",
        "textattack/bert-base-uncased-SST-2",
        "textattack/xlnet-base-cased-SST-2",
        "textattack/albert-base-v2-SST-2",
        "siebert/sentiment-roberta-large-english",
        "textattack/distilbert-base-cased-SST-2",
        "textattack/xlnet-large-cased-SST-2"
];

local models_and_experiments = [
    Experiments(
        BASE_PATH,
        LANGUAGE,
        m
    ) for m in models
];

local flat_models_and_experiments = [
    item for sublist in models_and_experiments for item in sublist
];

local config = flat_models_and_experiments;

{
    [std.format("%d.json", i)]: config[i] for i in std.range(0, std.length(config)-1)
}