import unittest
from unittest.case import expectedFailure

class TestModels(unittest.TestCase):

    def test_gpt2(self):
        from word_segmentation.beamsearch.gpt2_lm import GPT2LM
        model_name_or_path = 'gpt2'

        model = GPT2LM(
            model_name_or_path, 
            device='cuda',
            gpu_batch_size=1)
        scores = model.get_probs(
            [
                'h elloworld',
                'he lloworld',
                'hel loworld',
                'hello world'
            ]
        )
        expected_scores = [
            37.71688461303711, 
            41.082313537597656, 
            44.12258529663086, 
            23.88399887084961
        ]
        self.assertEqual(scores, expected_scores)
    
    def test_bert(self):
        from word_segmentation.beamsearch.bert_lm import BertLM
        model_name_or_path = 'bert-base-uncased'
        model = BertLM(
            model_name_or_path,
            device='cuda',
            gpu_batch_size=1
        )
        scores = model.get_probs(
            [
                'h elloworld',
                'he lloworld',
                'hel loworld',
                'hello world'
            ]
        )
        expected_scores = [
            46.93171286582947, 
            39.56816244125366, 
            53.853381633758545, 
            20.932902336120605
        ]
        self.assertEqual(scores, expected_scores)


if __name__ == '__main__':
    unittest.main()