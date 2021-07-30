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
        layers
    ) = {
        do_eval: true,
        log_level: "INFO",
        dataset_load_path: "./semeval2017",
        split: "test",
        run_classifier: true,
        sentiment_model: model,
        sentiment_model_device: 0,
        batch_size: 6,
        metrics: "./sentiment_metrics.py",
        prune_layers: layers,
        run_segmenter: false
    };

local config = [
    Params("cardiffnlp/twitter-roberta-base-sentiment", [6, 12]),
    Params("finiteautomata/bertweet-base-sentiment-analysis", [6, 12]),
    Params("distilbert-base-uncased-finetuned-sst-2-english", [6, 12]),
    Params("textattack/roberta-base-SST-2", [6, 12]),
    Params("textattack/bert-base-uncased-SST-2", [6, 12]),
    Params("textattack/xlnet-base-cased-SST-2", [6, 12]),
    Params("textattack/albert-base-v2-SST-2", [6, 12]),
    Params("textattack/facebook-bart-large-SST-2", [6, 12]),
    Params("textattack/distilbert-base-uncased-SST-2", [6]),
    Params("textattack/distilbert-base-cased-SST-2", [6]),
    Params("textattack/xlnet-large-cased-SST-2", [6, 12, 18, 24]),
    Params("siebert/sentiment-roberta-large-english", [6, 12, 18, 24])
];

{
    [std.format("%d.json", i)]: config[i] for i in std.range(0, std.length(config)-1)
}