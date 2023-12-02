import clip
from pathlib import Path
import os
import sys
import torch
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
# from prompt_toolkit import prompt
import pandas as pd
from tqdm import tqdm
from WavCaps.retrieval.models.ase_model import ASE
from ruamel import yaml


def zeroshot_classifier(classnames, templates, device):
    with torch.no_grad():
        zeroshot_weights = {}
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).to(device) #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights[classname] = class_embedding.cpu().detach().numpy().astype(np.float32)
        # zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


df = pd.read_csv('/home/aoq234/thesis/ClipClap-GZSL/avgzsl_benchmark_non_averaged_datasets/UCF/class-split/ucf_clip_class_names.csv')
ucf_classes = df['clip_class_name'].tolist()

ucf_templates = [
    'a photo of a person {}.',
    'a video of a person {}.',
    'a example of a person {}.',
    'a demonstration of a person {}.',
    'a photo of the person {}.',
    'a video of the person {}.',
    'a example of the person {}.',
    'a demonstration of the person {}.',
    'a photo of a person using {}.',
    'a video of a person using {}.',
    'a example of a person using {}.',
    'a demonstration of a person using {}.',
    'a photo of the person using {}.',
    'a video of the person using {}.',
    'a example of the person using {}.',
    'a demonstration of the person using {}.',
    'a photo of a person doing {}.',
    'a video of a person doing {}.',
    'a example of a person doing {}.',
    'a demonstration of a person doing {}.',
    'a photo of the person doing {}.',
    'a video of the person doing {}.',
    'a example of the person doing {}.',
    'a demonstration of the person doing {}.',
    'a photo of a person during {}.',
    'a video of a person during {}.',
    'a example of a person during {}.',
    'a demonstration of a person during {}.',
    'a photo of the person during {}.',
    'a video of the person during {}.',
    'a example of the person during {}.',
    'a demonstration of the person during {}.',
    'a photo of a person performing {}.',
    'a video of a person performing {}.',
    'a example of a person performing {}.',
    'a demonstration of a person performing {}.',
    'a photo of the person performing {}.',
    'a video of the person performing {}.',
    'a example of the person performing {}.',
    'a demonstration of the person performing {}.',
    'a photo of a person practicing {}.',
    'a video of a person practicing {}.',
    'a example of a person practicing {}.',
    'a demonstration of a person practicing {}.',
    'a photo of the person practicing {}.',
    'a video of the person practicing {}.',
    'a example of the person practicing {}.',
    'a demonstration of the person practicing {}.',
]



device = 'cuda:3'
model, preprocess = clip.load("ViT-B/32", device=device)



model = model.to(device)
model.eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)





zeroshot_weights = zeroshot_classifier(ucf_classes, ucf_templates, device)
print(zeroshot_weights.keys())
data_root_path = '/home/aoq234/akata-shared/aoq234/avzsl/clip_original/avgzsl_benchmark_datasets/UCF/features/cls_features_non_averaged'
data_path = os.path.join(data_root_path, 'text')

if not(os.path.exists(data_path)):
    os.makedirs(data_path)
filename = os.path.join(data_path, 'word_embeddings_ucf_normed.npy')


np.save(filename, zeroshot_weights)















def wavcaps_zeroshot_classifier(classnames, templates, device):
    with torch.no_grad():
        zeroshot_weights = {}
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class

            class_embeddings = wavcaps_model.encode_text(texts) #embed with text encoder
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights[classname] = class_embedding.cpu().detach().numpy().astype(np.float32)
    return zeroshot_weights





ucf_audio_templates = [
    'a person {} can be heard.',
    'a example of a person {} can be heard.',
    'a demonstration of a person {} can be heard.',
    'the person {} can be heard.',
    'a example of the person {} can be heard.',
    'a demonstration of the person {} can be heard.',
    'a person using {} can be heard.',
    'a example of a person using {} can be heard.',
    'a demonstration of a person using {} can be heard.',
    'a example of the person using {} can be heard.',
    'a demonstration of the person using {} can be heard.',
    'a person doing {} can be heard.',
    'a example of a person doing {} can be heard.',
    'a demonstration of a person doing {} can be heard.',
    'a example of the person doing {} can be heard.',
    'a demonstration of the person doing {} can be heard.',
    'a example of a person during {} can be heard.',
    'a demonstration of a person during {} can be heard.',
    'a example of the person during {} can be heard.',
    'a demonstration of the person during {} can be heard.',
    'a person performing {} can be heard.',
    'a example of a person performing {} can be heard.',
    'a demonstration of a person performing {} can be heard.',
    'a example of the person performing {} can be heard.',
    'a demonstration of the person performing {} can be heard.',
    'a person practicing {} can be heard.',
    'a example of a person practicing {} can be heard.',
    'a demonstration of a person practicing {} can be heard.',
    'a example of the person practicing {} can be heard.',
    'a demonstration of the person practicing {} can be heard.'
]


with open("/home/aoq234/dev/CLIP-GZSL/WavCaps/retrieval/settings/inference.yaml", "r") as f:
    config = yaml.safe_load(f)
device = 'cuda:3'
wavcaps_model = ASE(config)
wavcaps_model.to(device)

cp_path = '/home/aoq234/dev/CLIP-GZSL/WavCaps/retrieval/pretrained_models/audio_encoders/HTSAT_BERT_zero_shot.pt'
state_dict_key = 'model'
cp = torch.load(cp_path)
wavcaps_model.load_state_dict(cp[state_dict_key])
wavcaps_model.eval()
print("Model weights loaded from {}".format(cp_path))

wavecaps_zeroshot_weights = wavcaps_zeroshot_classifier(ucf_classes, ucf_audio_templates, device)


print(wavecaps_zeroshot_weights.keys())
data_path = os.path.join(data_root_path, 'text')

if not(os.path.exists(data_path)):
    os.makedirs(data_path)
filename = os.path.join(data_path, 'wavcaps_word_embeddings_ucf_normed.npy')


np.save(filename, wavecaps_zeroshot_weights)


# python clip_embeddings_extraction/get_clip_embeddings_ucf.py
