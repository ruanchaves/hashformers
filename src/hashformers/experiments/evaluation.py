from hashformers.evaluation.modeler import Modeler
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

def filter_top_k(
    input_df, 
    k, 
    gold_field="hashtag", 
    score_field="score",
    segmentation_field="segmentation",
    fill=False):
  """
  Filters the top k rows of the input_df for each group defined by the gold_field. 

  The function sorts the input_df by score_field in ascending order and retains the first k rows for each group.
  If fill option is set to True, it also clones the records based on the length of each group. 

  Args:
      input_df (pandas.DataFrame): The input dataframe to filter.
      k (int): The number of top records to retain for each group.
      gold_field (str, optional): The field used to define groups in the dataframe. Defaults to "hashtag".
      score_field (str, optional): The field used to sort the dataframe. Defaults to "score".
      segmentation_field (str, optional): The field used if the fill option is set to True. Defaults to "segmentation".
      fill (bool, optional): Whether to clone the records based on the length of each group. Defaults to False.

  Returns:
      pandas.DataFrame: The filtered dataframe.
  """
  df = copy.deepcopy(input_df)
  
  df = df\
    .sort_values(by=score_field, ascending=True)\
    .groupby(gold_field)\
    .head(k)

  if fill:
    df["group_length"] = df.groupby(gold_field)[segmentation_field].transform(len)
    df["group_length"] = df["group_length"] * -1 + k + 1
    len_array = df["group_length"].values
    
    df = df.drop(columns=["group_length"])
    records = np.array(df.to_dict("records"))
    cloned_records = list(np.repeat(records, len_array))
    df = pd.DataFrame(cloned_records)
    
    df = df\
      .sort_values(by=score_field, ascending=True)\
      .groupby(gold_field)\
      .head(k)

    length = df.groupby(gold_field).size().values
    assert (length == k).all()
  
  return df


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