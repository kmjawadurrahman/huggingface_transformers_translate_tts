import torch
from transformers import BarkModel, AutoProcessor


device = "cuda:0" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")

model = model.to(device)

inputs = processor("Comment vous appelez-vous ?", voice_preset="fr_speaker_3")

speech_output = model.generate(**inputs.to(device))

print(speech_output)