from word_segmentation.beamsearch.data_structures import enforce_prob_dict
from word_segmentation.evaluation.modeler import Modeler

def evaluate_dictionary(data, gold, n=11):
    
    gold_dict = {}
    for item in gold:
        gold_dict.update({item.replace(" ", ""): item})

    input_data = enforce_prob_dict(data)

    final_metrics = {}
    for i in range(1, n):
        df = input_data.get_top_k(
            self,
            k=i,
            characters_field="characters",
            segmentation_field="segmentation",
            score_field="score",
            return_dataframe=True
        )

        df['gold'] = df['characters'].apply(
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
             records['gold'],
             records['segmentation']   
            )
        
        metrics = {
                'f1': modeler.calculateFScore(),
                'accuracy': modeler.calculateAccuracy(),
                'precision': modeler.calculatePrecision(),
                'recall': modeler.calculateRecall()
            }

        if i > 1:
            metrics = {
                f"top_{i}_{key}" for key, value in metrics.items()
            }

        final_metrics.update(metrics)
    
    return final_metrics