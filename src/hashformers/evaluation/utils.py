from hashformers.beamsearch.data_structures import enforce_prob_dict
from hashformers.evaluation.modeler import Modeler

def evaluate_dictionary(data, gold, n=10):
    """
    Evaluates the input data dictionary against a gold standard using various metrics. 
    It computes these metrics for the top 'n' entries of the data.

    Args:
        data (dict or ProbabilityDictionary): The input data to be evaluated.
        gold (list): The gold standard list of strings to compare against.
        n (int, optional): The number of top entries from the data to be considered for evaluation. Default is 10.

    Returns:
        dict: A dictionary containing the computed metrics (F1 score, accuracy, precision, recall) for each of the top 'n' entries.
    """
    gold_dict = {}
    for item in gold:
        gold_dict.update({item.replace(" ", ""): item})

    input_data = enforce_prob_dict(data)

    final_metrics = {}
    for i in range(1, n+1):
        df = input_data.get_top_k(
            k=i,
            characters_field="hashtag",
            segmentation_field="segmentation",
            score_field="score",
            return_dataframe=True
        )

        df['gold'] = df['hashtag'].apply(
            lambda x: gold_dict[x]
        )

        if i > 1:
            df['truth_field'] = df['gold'].combine(
                df['segmentation'],
                lambda x,y: int(x == y)
            )
            df = df.sort_values(
                by='truth_field',
                ascending=False)
            df = df.groupby('gold').head(1)

        records = df.to_dict('records')
        modeler = Modeler()
        for item in records:
            modeler.countEntry(
             item['gold'],
             item['segmentation']   
            )
        
        metrics = {
                'f1': modeler.calculateFScore(),
                'accuracy': modeler.calculateAccuracy(),
                'precision': modeler.calculatePrecision(),
                'recall': modeler.calculateRecall()
            }

        if i > 1:
            metrics = {
                f"top_{i}_{key}":value for key, value in metrics.items()
            }

        final_metrics.update(metrics)
    
    return final_metrics