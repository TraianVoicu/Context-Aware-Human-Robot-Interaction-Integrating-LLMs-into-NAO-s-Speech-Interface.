import sounddevice as sd
import numpy as np
import whisper
from tts_to_llama import trimite_catre_llama


duration = 3  
fs = 16000     
print("Vorbește acum...")
audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)     
sd.wait()


model = whisper.load_model("medium")  
result = model.transcribe(np.squeeze(audio), fp16=False, language="en")
text = result["text"].strip()
print("Ai zis:", text)
cod_actiune = trimite_catre_llama(text)
print("Cod acțiune primit:", cod_actiune)
