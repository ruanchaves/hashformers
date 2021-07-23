local Params(
    model,
    layers,
    negative,
    neutral,
    positive,
    skip_neutral
) = {
    do_eval: true,
    log_level: "INFO",
    dataset_load_path: "./semeval2017",
    split: "test",
    hashtag_only: true,
    negative_field: negative,
    neutral_field: neutral,
    positive_field: positive,
    run_classifier: true,
    sentiment_model: model,
    sentiment_model_device: 0,
    batch_size: 1,
    metrics: "./sentiment_metrics.py",
    prune_layers: layers,
    run_segmenter: false,
    skip_neutral: skip_neutral
 };

{
    "1.json": Params("cardiffnlp/twitter-roberta-base-sentiment", 12, "0", "1", "2", false),
    "2.json": Params("cardiffnlp/twitter-xlm-roberta-base-sentiment", 12, "0", "1", "2", false),
    "3.json": Params("finiteautomata/bertweet-base-sentiment-analysis", 12, "0", "1", "2", false)
}