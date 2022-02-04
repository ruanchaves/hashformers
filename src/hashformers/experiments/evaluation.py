from hashformers.evaluation.modeler import Modeler
import pandas as pd
import copy
import numpy as np

def evaluate_df(
    df, 
    gold_field="gold", 
    segmentation_field="segmentation"
):
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
    selected_data = [ 
      x for x in data if x["dataset"]==dataset \
        and x["model"]==model
      ][0]["data"]
    output = pd.DataFrame(selected_data)
    return output