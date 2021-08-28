# Hashtag Segmentation

This example is meant for the **evaluation** of hashtag segmentation pipelines.

`hashtag_segmentation.py` takes a list of segmented hashtags as its `source` argument. The `Test-BOUN` file shows the expected input format for a source file:

```
...
friday night 
Ryan Braun 
Update 
Hotel 
snake skin 
cast 
stand united 
...
```

All hashtags will be rejoined and segmented again by the chosen models. Performance metrics will be logged at the end of the evaluation.

## Usage

```
python hashtag_segmentation.py \
    --source ./Test-BOUN \
    --evaluate True \
    --evaluate_top_k 10 \
    --output_dir ./output \
    --save_to_output_dir True \
    --segmenter_results_filename segmenter.csv \
    --reranker_results_filename reranker.csv \
    --ensemble_results_filename ensemble.csv \
    --segmentation_results_filename segmentations.txt
```