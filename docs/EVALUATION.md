# Evaluation

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