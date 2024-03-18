import gradio as gr
import torch
from transformers import pipeline, BarkModel, AutoProcessor


device = "cuda:0" if torch.cuda.is_available() else "cpu"

translator = pipeline(task="translation",
                      model="facebook/nllb-200-distilled-600M",
                      torch_dtype=torch.bfloat16)

processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")

sampling_rate = model.generation_config.sample_rate

model = model.to(device)

def launch(input_text):
    translated_french_text = translator(input_text,
                                    src_lang="eng_Latn",
                                    tgt_lang="fra_Latn")
    
    speech_input = processor(translated_french_text[0]["translation_text"],
                       voice_preset="fr_speaker_3")

    speech_output_tensor = model.generate(**speech_input.to(device))

    speech_output = speech_output_tensor[0].cpu().numpy()

    return translated_french_text[0]["translation_text"], (sampling_rate, speech_output)

interface = gr.Interface(launch,
                         inputs="text",
                         outputs=["text", gr.Audio()])

interface.launch()