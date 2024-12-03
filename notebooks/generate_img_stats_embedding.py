import torch
import clip
from PIL import Image

train_image_names_path = (
    "/data/user_data/sbharad2/SpeechCLIP/data/flickr/Flickr_8k.trainImages.txt"
)
with open(train_image_names_path) as f:
    train_image_names = f.readlines()
train_image_names = [x.strip() for x in train_image_names]

clip_model = "ViT-B/32"
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(clip_model, device=device)

all_image_features = None
c = 0
with torch.no_grad():
    for image_name in train_image_names:
        image = (
            preprocess(
                Image.open(
                    f"/data/user_data/sbharad2/SpeechCLIP/data/flickr/Images/{image_name}"
                )
            )
            .unsqueeze(0)
            .to(device)
        )
        image_features = model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        if all_image_features is None:
            all_image_features = image_features
        else:
            all_image_features += image_features
        c += 1
        if c % 100 == 0:
            print(c)

all_image_features = all_image_features / c
all_image_features = all_image_features / all_image_features.norm(dim=-1, keepdim=True)

import numpy as np

save_path = f"/data/user_data/sbharad2/SpeechCLIP/data/flickr_stats/{clip_model.replace('/','_').replace('-','_')}.npy"
np.save(save_path, all_image_features.squeeze(0).cpu().numpy())
print((all_image_features**2).sum(dim=-1))
