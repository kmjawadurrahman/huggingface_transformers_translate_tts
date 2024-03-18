import torch
from transformers import pipeline


translator = pipeline(task="translation",
                      model="facebook/nllb-200-distilled-600M",
                      torch_dtype=torch.bfloat16)

english_text = "What is you name?"

translated_french_text = translator(english_text,
                                    src_lang="eng_Latn",
                                    tgt_lang="fra_Latn")

print(translated_french_text[0]["translation_text"])