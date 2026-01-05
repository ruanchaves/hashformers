from hashformers.evaluation.modeler import Modeler
from hashformers.utils.filtering import filter_top_k
import pandas as pd
import copy
import numpy as np

def evaluate_df(
    df, 
    gold_field="gold", 
    segmentation_field="segmentation"
):
  """
  Evaluates the given dataframe based on the gold_field and segmentation_field and returns various metric values.

  This function creates a new column "truth_value" in the dataframe by comparing gold_field and segmentation_field.
  It then sorts the dataframe by gold_field and "truth_value" and retains the first row for each gold_field group.
  The metrics calculated include F1 score, accuracy, recall and precision.

  Args:
      df (pandas.DataFrame): The dataframe to be evaluated.
      gold_field (str, optional): The field in the dataframe used as the 'truth' field for evaluation. Defaults to "gold".
      segmentation_field (str, optional): The field in the dataframe used as the 'prediction' field for evaluation. Defaults to "segmentation".
  
  Returns:
      dict: A dictionary containing F1 score, accuracy, recall, and precision metrics.
  """
  evaluator = Modeler()

  df["truth_value"] = df[gold_field].combine(
    df[segmentation_field],
    lambda x,y: x == y
  )

  df = df\
    .sort_values(
      by=[
        gold_field, 
        "truth_value"
      ], 
      ascending=False)\
    .groupby(gold_field)\
    .head(1)

  records = df.to_dict("records")
  for row in records:
    evaluator.countEntry(
      row[segmentation_field],
      row[gold_field]
    )
  metrics = {
      "f1": evaluator.calculateFScore(),
      "acc": evaluator.calculateAccuracy(),
      "recall": evaluator.calculateRecall(),
      "precision": evaluator.calculatePrecision()
  }
  return metrics

def read_experiment_dataset(data, dataset, model):
    """
    Reads and returns the dataset for a given model from a collection of datasets.

    The function filters the data based on the dataset and model parameters, converts the filtered data into a pandas 
    DataFrame and returns it.

    Args:
        data (list of dicts): The collection of datasets. Each element is a dictionary which must contain "dataset" and 
        "model" keys along with a "data" key which contains the actual data.
        dataset (str): The name of the dataset to read.
        model (str): The name of the model for which the dataset needs to be read.

    Returns:
        pandas.DataFrame: The selected dataset as a dataframe.
    """
    selected_data = [ 
      x for x in data if x["dataset"]==dataset \
        and x["model"]==model
      ][0]["data"]
    output = pd.DataFrame(selected_data)
    return output