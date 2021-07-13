import string
import numpy as np

def clear_formatting(entry):
    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    entry = entry.translate(translator)
    entry = entry.lower()
    return entry

def calculate_recall(prediction, reference):
        prediction = clear_formatting(prediction).split(" ")
        reference = clear_formatting(reference).split(" ")
        
        total_length = len(reference)
        
        calculate_length = lambda x: np.cumsum([len(y) for y in x])
        prediction_length = calculate_length(prediction)
        reference_length = calculate_length(reference)
        
        prediction_tuples = list(zip(prediction, prediction_length))
        reference_tuples = list(zip(reference, reference_length))
        
        intersection_set = set(reference_tuples).intersection(prediction_tuples)
        TP = len(intersection_set)
        return float(TP) / float(total_length)

def calculate_precision(prediction, reference):
        prediction = clear_formatting(prediction).split(" ")
        reference = clear_formatting(reference).split(" ")
                
        calculate_length = lambda x: np.cumsum([len(y) for y in x])
        prediction_length = calculate_length(prediction)
        reference_length = calculate_length(reference)
        
        prediction_tuples = list(zip(prediction, prediction_length))
        reference_tuples = list(zip(reference, reference_length))
        
        intersection_set = set(reference_tuples).intersection(prediction_tuples)
        difference_set = set(prediction_tuples).difference(reference_tuples)
        TP = len(intersection_set)
        FP = len(difference_set)
        return float(TP) / (float(FP) + float(TP)) if (float(FP) + float(TP)) else 0

def calculate_f1(prediction, reference):
        prediction = clear_formatting(prediction).split(" ")
        reference = clear_formatting(reference).split(" ")
        
        total_length_precision = len(prediction)
        total_length_recall = len(reference)
        
        calculate_length = lambda x: np.cumsum([len(y) for y in x])
        prediction_length = calculate_length(prediction)
        reference_length = calculate_length(reference)
        
        prediction_tuples = list(zip(prediction, prediction_length))
        reference_tuples = list(zip(reference, reference_length))
        intersection_set = set(reference_tuples).intersection(prediction_tuples)
        difference_set = set(prediction_tuples).difference(reference_tuples)

        TP = len(intersection_set)
        FP = len(difference_set)
        recall = float(TP) / float(total_length_recall)
        precision = float(TP) / (float(FP) + float(TP)) if (float(FP) + float(TP)) else 0
        recall_1 = 1/recall if recall else 0
        precision_1 = 1/precision if precision else 0
        sum_1 = recall_1 + precision_1
        fact = 1/sum_1 if sum_1 else 0
        return 2*(fact)