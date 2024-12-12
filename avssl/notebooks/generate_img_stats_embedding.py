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
            all_image_features = image_features.clone()
        else:
            all_image_features += image_features.clone()
        c += 1
        if c % 100 == 0:
            print(c)


all_image_features = all_image_features / c
all_image_features = all_image_features / all_image_features.norm(dim=-1, keepdim=True)

import numpy as np

all_sim = []

# for i in range(len(all_text_features)):
#     for j in range(len(all_text_features)):
#         if i == j:
#             continue
#         sim = all_text_features[i] @ all_text_features[j].T
#         if sim.item() > 1:
#             print(i, j)
#             print(all_text_features[i].shape, all_text_features[j].shape)
#             print(sim.item())
#             print((all_text_features[i] ** 2).sum(dim=-1))
#             print((all_text_features[j] ** 2).sum(dim=-1))
#             break
#         all_sim.append(sim.item())

# print(sum(all_sim) / len(all_sim))
# print(min(all_sim), max(all_sim))

save_path = f"/data/user_data/sbharad2/SpeechCLIP/data/flickr_stats/cloned.img_stats.{clip_model.replace('/','_').replace('-','_')}.npy"
np.save(save_path, all_image_features.squeeze(0).cpu().numpy())
print((all_image_features**2).sum(dim=-1))
