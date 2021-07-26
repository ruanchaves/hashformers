local WS_Params(
        
    ) = {
        log_level: "INFO",
        run_classifier: false,
        run_segmenter: true,
        dataset_reader: "./semeval2017.py",
        dataset_save_path: "./semeval2017",
        dataset_url: "",
        split: "test",
        hashtag_only: true,
        
    };

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

local config = [
    Params("cardiffnlp/twitter-roberta-base-sentiment", 12, "0", "1", "2", false),
    Params("cardiffnlp/twitter-xlm-roberta-base-sentiment", 12, "0", "1", "2", false),
    Params("finiteautomata/bertweet-base-sentiment-analysis", 12, "0", "1", "2", false),
    Params("distilbert-base-uncased-finetuned-sst-2-english", 12, "0", "2", "1", true),
    Params("textattack/roberta-base-SST-2", 12, "0", "2", "1", true),
    Params("textattack/bert-base-uncased-SST-2", 12, "0", "2", "1", true),
    Params("textattack/xlnet-base-cased-SST-2", 12, "0", "2", "1", true),
    Params("textattack/albert-base-v2-SST-2", 12, "0", "2", "1", true),
    Params("textattack/facebook-bart-large-SST-2", 12, "0", "2", "1", true),
    Params("textattack/distilbert-base-uncased-SST-2", 6, "0", "2", "1", true),
    Params("textattack/xlnet-large-cased-SST-2", 24, "0", "2", "1", true),
    Params("textattack/distilbert-base-cased-SST-2", 6, "0", "2", "1", true)
];

{
    [std.format("%d.json", i)]: config[i] for i in std.range(0, std.length(config)-1)
}