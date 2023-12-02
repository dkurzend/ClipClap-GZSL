import clip
from PIL import Image

from pathlib import Path
import os
import sys
sys.path.append("..")
from tqdm import tqdm
import torch
from c3d.c3d import C3D
import torchvision
import csv
from csv import reader
import numpy as np
from timeit import default_timer as timer
from audioset_vggish_tensorflow_to_pytorch.vggish import VGGish
from audioset_vggish_tensorflow_to_pytorch.audioset import vggish_input, vggish_postprocess
import cv2
from pydub import AudioSegment
import pickle
from ruamel import yaml
import glob
import soundfile as sf
import librosa
import torch.nn.functional as F
from WavCaps.retrieval.models.ase_model import ASE
import argparse
from src.args import str_to_bool


def read_prepare_audio(audio_path, device):
    audio, _ = librosa.load(audio_path, sr=32000, mono=True)
    audio = torch.tensor(audio).unsqueeze(0).to(device)
    # pad
    if audio.shape[-1] < 32000 * 10:
        pad_length = 32000 * 10 - audio.shape[-1]
        audio = F.pad(audio, [0, pad_length], "constant", 0.0)
    # crop
    elif audio.shape[-1] > 32000 * 10:
        center = audio.shape[-1] // 2
        start  = center - (32000 * 5)
        end  = center + (32000 * 5)
        audio = audio[:, start:end]
    return audio



parser = argparse.ArgumentParser(description='Create the datasets.')
parser.add_argument('--finetuned_model', type=str_to_bool, default=True,
                    help='Whether to load original Clip and WavCaps models or the finetuned verisons')


args = parser.parse_args()



device = 'cuda:4'
model, preprocess = clip.load("ViT-B/32", device=device)
if args.finetuned_model == True:
    model_path = '/home/aoq234/dev/ClipClap/logs/clip_finetuning/second_try_Aug25_19-53-14_475888_callisto/checkpoints/clip_finetuned.pt'
    save_path = '/home/aoq234/akata-shared/aoq234/mnt/ucf_features_finetuned_clip_wavcaps'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model_path = "ViT-B/32"
    save_path = '/home/aoq234/akata-shared/aoq234/mnt/ucf_features_original_clip_wavcaps'


model = model.to(device)
model.eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)







output_list_no_average=[]


path=Path("/home/aoq234/akata-shared/datasets/UCF101/UCF-101") # path to search for videos

dict_csv={}
list_classes=[]
count=0
dict_classes_ids={}


for f in tqdm(path.glob("**/*.avi")):
    class_name=str(f).split("/")[-2]
    if class_name not in list_classes:
        list_classes.append(class_name)

list_classes.sort()

for index,val in enumerate(sorted(list_classes)):
    dict_classes_ids[val]=index



with open("/home/aoq234/dev/CLIP-GZSL/WavCaps/retrieval/settings/inference.yaml", "r") as f:
    config = yaml.safe_load(f)



wavcaps_model = ASE(config)
wavcaps_model.to(device)

if args.finetuned_model == True:
    cp_path = '/home/aoq234/dev/ClipClap/logs/wavcaps_finetuning/first_try_Aug29_07-25-05_613403_callisto/checkpoints/WavCaps_finetuned.pt'
    state_dict_key = 'model_state_dict'
else:
    cp_path = '/home/aoq234/dev/CLIP-GZSL/WavCaps/retrieval/pretrained_models/audio_encoders/HTSAT_BERT_zero_shot.pt'
    state_dict_key = 'model'

cp = torch.load(cp_path)
wavcaps_model.load_state_dict(cp[state_dict_key])
wavcaps_model.eval()
print("Model weights loaded from {}".format(cp_path))



counter=0

for f in tqdm(path.glob("**/*.avi")):

    counter+=1

    if counter%3000==0:
        with open(save_path, 'wb') as handle:
            pickle.dump(output_list_no_average, handle, protocol=pickle.HIGHEST_PROTOCOL)


    try:
        # audio
        mp4_version = AudioSegment.from_file(str(f), "avi")
        mp4_version.export("/home/aoq234/akata-shared/aoq234/mnt/ucf_dummy_tmp.wav", format="wav")
        audio = read_prepare_audio("/home/aoq234/akata-shared/aoq234/mnt/ucf_dummy_tmp.wav", device)

        with torch.no_grad():
            audio_emb = wavcaps_model.encode_audio(audio).squeeze()
        audio_emb = audio_emb.cpu().detach().numpy() # (1024,)




        cap = cv2.VideoCapture(str(f))
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video = torch.zeros((frameCount, frameHeight,frameWidth, 3), dtype=torch.float32)  # torch.Size([164, 240, 320, 3])

        fc = 0
        ret = True

        while (fc < frameCount and ret):
            ret, image=cap.read()
            if ret==True:
                torch_image=torch.from_numpy(image)
                video[fc]=torch_image
                fc += 1

        cap.release()


        list_clips=[]

        p= torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage() # requires tensor of shape  C x H x W
        ])


        mid_frame_idx = video.shape[0] // 2
        frame = video[mid_frame_idx]
        frame=frame.permute(2, 0, 1)
        frame=p(frame)

        # preprocess() calls the toTensor transformation which returns a  torch.FloatTensor of shape (C x H x W)
        # returns shape  (1, 3, 224, 224) = (bs, channels, height, width)
        frame = preprocess(frame).unsqueeze(0).to(device) # preprocess and add batch dimension

        with torch.no_grad():
            image_features = model.encode_image(frame) # shape torch.Size([1, 512])
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.squeeze()
        image_features=image_features.cpu().detach().numpy() # shape (512,)





    except Exception as e:
        print(e)
        print(f)
        continue


    name_file=str(f).split("/")[-1]
    class_name=str(f).split("/")[-2]
    class_id=dict_classes_ids[class_name]

    result_list=[image_features, class_id, audio_emb, name_file]



    output_list_no_average.append(result_list)



with open(save_path, 'wb') as handle:
    pickle.dump(output_list_no_average, handle, protocol=pickle.HIGHEST_PROTOCOL)





# python clip_feature_extraction/get_clip_features_ucf.py --finetuned_model False
