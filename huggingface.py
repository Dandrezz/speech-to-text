from transformers import Wav2Vec2Processor, HubertForCTC
import soundfile as sf
from pydub import AudioSegment 
from pydub.utils import make_chunks 
import torch
import os

processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")


myaudio = AudioSegment.from_file("podcast.wav", "wav") 
chunk_length_ms = 28000
chunks = make_chunks(myaudio,chunk_length_ms)
for i, chunk in enumerate(chunks): 
    chunk_name = "split/{0}.wav".format(i) 
    print ("exporting", chunk_name) 
    chunk.export(chunk_name, format="wav")

for i in sorted(os.listdir('./split')):
  speech, _ = sf.read('split/{}'.format(i))
  input_values = processor(speech, return_tensors="pt", sampling_rate=16000).input_values.to("cpu")
  logits = model(input_values).logits
  predicted_ids = torch.argmax(logits, dim=-1)
  transcription = processor.decode(predicted_ids[0])
  print(transcription)
