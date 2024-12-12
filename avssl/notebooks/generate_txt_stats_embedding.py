import torch
import clip
from PIL import Image

train_image_names_path = (
    "/data/user_data/sbharad2/SpeechCLIP/data/flickr/Flickr_8k.trainImages.txt"
)
all_text_paths = "/data/user_data/sbharad2/SpeechCLIP/data/flickr/Flickr8k.token.txt"  # Make sure to filter out test set from this.

with open(train_image_names_path) as f:
    train_image_names = f.readlines()
train_image_names = set([x.strip() for x in train_image_names])

texts = []
with open(all_text_paths) as f:
    for line in f:
        img_id, text = line.strip().split("\t")
        img_id = img_id.split("#")[0]
        if img_id in train_image_names:
            texts.append(text.strip())
assert len(texts) == (len(train_image_names) * 5), (
    len(texts),
    len(train_image_names),
    "Mismatch in number of texts and images",
)
print(len(texts), "texts loaded")

clip_model = "ViT-L/14"  # "ViT-B/32"
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(clip_model, device=device)

all_text_features = None
c = 0
with torch.no_grad():
    for text in texts:
        text = clip.tokenize([text]).to(device)
        text_features = model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        if all_text_features is None:
            all_text_features = text_features.clone()
        else:
            all_text_features += text_features.clone()
        c += 1
        if c % 100 == 0:
            print(c)

all_text_features = all_text_features / c
all_text_features = all_text_features / all_text_features.norm(dim=-1, keepdim=True)


import numpy as np

save_path = f"/data/user_data/sbharad2/SpeechCLIP/data/flickr_stats/cloned.text_stats.CLIP_{clip_model.replace('/','_').replace('-','_')}.npy"
np.save(save_path, all_text_features.squeeze(0).cpu().numpy())
print((all_text_features**2).sum(dim=-1), c)
