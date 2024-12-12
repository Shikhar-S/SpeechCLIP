import torch
import whisper
import pandas as pd
import numpy as np
from avssl.model.kwClip import KWClip_SpeechText

df_path = "/data/user_data/sbharad2/SpeechCLIP/data/flickr/flickr.csv"
df = pd.read_csv(df_path)  # example_id, caption, wav, split, image_id
print("All data:", df.shape)
df = df[df["split"] == "train"]
print("Train data:", df.shape)


device = "cuda" if torch.cuda.is_available() else "cpu"

# load model
ckpt_path = "/compute/babel-11-13/sbharad2/m3l_runs/t.linear_proj.20241201.133909/epoch=208-step=24452-val_recall_mean_10=69.9097.ckpt"
model = None
model = KWClip_SpeechText.load_from_checkpoint(ckpt_path)
# model = model.half()
model.to(device)

all_speech_features = None
c = 0
with torch.no_grad():
    for wav_name in df["wav"]:
        audio_path = (
            "/data/user_data/sbharad2/SpeechCLIP/data/flickr/flickr_audio/wavs/"
            + wav_name
        )
        try:
            waveform = torch.from_numpy(whisper.load_audio(audio_path)).to(
                device=device
            )
            waveform = waveform.unsqueeze(0)
            wavlen = waveform.shape[-1]
            speech_features = model._extract_speech_features(waveform, wavlen)

            if all_speech_features is None:
                all_speech_features = speech_features
            else:
                all_speech_features += speech_features
            c += 1
            if c % 100 == 0:
                print(c)
        except:
            print("Error:", audio_path)

all_speech_features = all_speech_features / c
all_speech_features = all_speech_features / all_speech_features.norm(
    dim=-1, keepdim=True
)

import numpy as np

save_path = f"/data/user_data/sbharad2/SpeechCLIP/data/flickr_stats/speech_stats.contrastive_e208.npy"
np.save(save_path, all_speech_features.squeeze(0).cpu().numpy())
print((all_speech_features**2).sum(dim=-1))
