local Params(model,layers,negative,neutral,positive,skip_neutral) = 
    local param(
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
    [param(model, x, negative, neutral, positive, skip_neutral) for x in std.range(1, layers)];

local config = [
    Params("cardiffnlp/twitter-roberta-base-sentiment", 12, "0", "1", "2", false),
    Params("cardiffnlp/twitter-xlm-roberta-base-sentiment", 12, "0", "1", "2", false),
    Params("finiteautomata/bertweet-base-sentiment-analysis", 12, "0", "1", "2", false)
];

local param_list = [ item for items in config for item in items];

{
    [std.format("%d.json", i)]: param_list[i] for i in std.range(0, std.length(param_list)-1)
}