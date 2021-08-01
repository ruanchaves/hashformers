from dataclasses import dataclass
from word_segmentation import WordSegmenter
import datasets

class Translator(WordSegmenter):

    def __init__(
        self,
        translation_model=None,
        translation_tokenizer=None,
        **kwargs
    ):
        self.translation_model = translation_model
        self.translation_tokenizer = translation_tokenizer
        super().__init__(**kwargs)

    def translate_sentence(self, sentence, device="cuda"):
        if isinstance(sentence, str):
            input = [sentence]
        else:
            input = sentence
        tokens = self.translation_tokenizer(input, return_tensors="pt", padding=True, truncation=True)
        tokens.to(device)
        translated = self.translation_model.generate(**tokens)
        translation = [self.translation_tokenizer.decode(t, skip_special_tokens=True) for t in translated]

        if isinstance(sentence, str):
            return translation[0]
        else:
            return translation 

    def translate(
        self,
        dataset,
        content_field="content",
        output_field="translation",
        **kwargs
    ):  

        translation_fn_kwargs = {
                    "model": self.translation_model,
                    "tokenizer": self.translation_tokenizer,
                    "content_field": content_field,
                    "output_field": output_field
        }

        def translate_row(
            row, 
            content_field="sentence",
            output_field="output",
            model=None, 
            tokenizer=None):
            row[output_field] = \
                self.translate_sentence(row[content_field], model=model, tokenizer=tokenizer)
            return row
        
        if isinstance(dataset, dict):
            json_object = {
                "key": [k for k in dataset.keys()],
                content_field: [v for v in dataset.values()]
            }
            converted_dataset = datasets.Dataset.from_dict(json_object)
            translated_dataset = converted_dataset.map(
                translate_row,
                fn_kwargs=translation_fn_kwargs,
                **kwargs
            )
            output_dataset_dict = translated_dataset.to_dict()
            output_dataset = {
                k:v for k,v in list(zip(
                    output_dataset_dict["key"],
                    output_dataset_dict[output_field]
                ))
            }
        else:
            output_dataset = dataset.map(
                translate_row,
                fn_kwargs=translation_fn_kwargs,
                **kwargs
            )

        return output_dataset