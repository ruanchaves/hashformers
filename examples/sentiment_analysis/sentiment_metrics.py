from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score
)

import datasets


_DESCRIPTION = """
"""

_KWARGS_DESCRIPTION = """
"""

_CITATION = """\
@article{scikit-learn,
  title={Scikit-learn: Machine Learning in {P}ython},
  author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
         and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
         and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
         Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
  journal={Journal of Machine Learning Research},
  volume={12},
  pages={2825--2830},
  year={2011}
}
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Accuracy(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                    "references": datasets.Value("string"),
                }
            ),
        )

    def _compute(self, predictions, references, average="macro"):
        return {
            "accuracy": accuracy_score(
                references, predictions
            ).item(),
            "recall": recall_score(
                references, predictions, average=average
            ).item(),
            "f1": f1_score(
                references, predictions, average=average
            ).item(),
            
        }