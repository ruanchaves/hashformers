# Evaluation

We provide a detailed evaluation of the accuracy and speed of the `hashformers` framework in comparison with alternative libraries.

Although models based on n-grams such as `ekphrasis` are orders of magnitude faster than `hashformers`, they are remarkably unstable across different domains. Research on word segmentation usually try to bring the best of both worlds together and combine deep learning with statistical methods for reaching the best speed-accuracy trade-off.

## Accuracy

<h1 align="center">
  <img src="https://raw.githubusercontent.com/ruanchaves/hashformers/master/barplot_evaluation.png" width="512" title="hashformers">
</h1>

In this figure we compare **hashformers** with [HashtagMaster](https://github.com/mounicam/hashtag_master) ( also known as "MPNR" ) and [ekphrasis](https://github.com/cbaziotis/ekphrasis) on five hashtag segmentation datasets.

HashSet-1 is a sample from the distant HashSet dataset. HashSet-2 is the lowercase version of HashSet-1, and HashSet-3 is the manually annotated portion of HashSet. More information on the datasets and their evaluation is available on the [HashSet paper](https://arxiv.org/abs/2201.06741). 

A script to reproduce the evaluation of ekphrasis is available on [scripts/evaluate_ekphrasis.py](https://github.com/ruanchaves/hashformers/blob/master/scripts/evaluate_ekphrasis.py).

| dataset       | library       |   accuracy |
|:--------------|:--------------|-----------:|
| BOUN          | HashtagMaster |     81.60  |
|               | ekphrasis     |     44.74  |
|               |**hashformers**|   **83.68**|
|               |               |            |
| HashSet-1     | HashtagMaster |     50.06  |
|               | ekphrasis     |      0.00  |
|               |**hashformers**|   **72.47**|
|               |               |            |
| HashSet-2     | HashtagMaster |     45.04  |
|               |**ekphrasis**  |   **55.73**|
|               | hashformers   |     47.43  |
|               |               |            |
| HashSet-3     | HashtagMaster |     41.93  |
|               | ekphrasis     |     56.44  |
|               |**hashformers**|   **56.71**|
|               |               |            |
| Stanford-Dev  | HashtagMaster |     73.12  |
|               | ekphrasis     |     51.38  |
|               |**hashformers**|   **80.04**|
|               |               |            |
| average (all) | HashtagMaster |     58.35  |
|               | ekphrasis     |     41.65  |
|               |**hashformers**|   **68.06**|

## Speed

| model         | hashtags/second | accuracy  | topk | layers|
|:--------------|:----------------|----------:|-----:|------:|
| ekphrasis     |    4405.00      |   44.74   |  -   |   -   |
| gpt2-large    |      12.04      |   63.86   |  2   | first |
| distilgpt2    |      29.32      |   64.56   |  2   | first |
|**distilgpt2** |    **15.00**    | **80.48** |**2** |**all**|
| gpt2          |      11.36      |    -      |  2   |  all  |
| gpt2          |      3.48       |    -      |  20  |  all  |
| gpt2 + bert   |      1.38       |   83.68   |  20  |  all  |

In this table we evaluate hashformers under different settings on the Dev-BOUN dataset and compare it with ekphrasis. As ekphrasis relies on n-grams, it is a few orders of magnitude faster than hashformers.  

All experiments were performed on Google Colab while connected to a Tesla T4 GPU with 15GB of RAM. We highlight `distilgpt2` at `topk = 2`, which provides the best speed-accuracy trade-off.

* **model**: The name of the model. We evaluate ekphrasis under the default settings, and use the reranker only for the SOTA experiment at the bottom row.

* **hashtags/second**: How many hashtags the model can segment per second. All experiments on hashformers had the `batch_size` parameter adjusted to take up close to 100% of GPU RAM. A sidenote: even at 100% of GPU memory usage, we get about 60% of GPU utilization. So you may get better results by using GPUs with more memory than 16GB.

* **accuracy**: Accuracy on the Dev-BOUN dataset. We don't evaluate the accuracy of `gpt2`, but we know [from the literature](https://arxiv.org/abs/2112.03213) that it is expected to be between `distilgpt2` (at 80%) and `gpt2 + bert` (the SOTA, at 83%).

* **topk**: the `topk` parameter of the Beamsearch algorithm ( passed as the `topk` argument to the `WordSegmenter.segment` method). The `steps` Beamsearch parameter was fixed at a default value of 13 for all experiments with hashformers, as it doesn't have a significant impact on performance as `topk`.

* **layers**: How many Transformer layers were utilized for language modeling: either all layers or just the bottom layer.