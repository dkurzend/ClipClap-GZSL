import clip
from PIL import Image

from pathlib import Path
import os
import sys
sys.path.append("..")
from tqdm import tqdm
import torch
import torchvision
import csv
from csv import reader
import numpy as np
from timeit import default_timer as timer
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
from memory_profiler import profile


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

parser.add_argument('--index', default=0, type=int)
parser.add_argument('--gpu', default=0, type=int)

args = parser.parse_args()



# device = 'cuda:4'
device = 'cuda:'+str(args.gpu)
model, preprocess = clip.load("ViT-B/32", device=device)

if args.finetuned_model == True:
    model_path = '/home/aoq234/dev/ClipClap/logs/activitynet_clip_finetuning/epoch1_lr06_activity_Sep14_13-30-51_901727_callisto/checkpoints/Clip_ActivityNet_finetuned.pt'
    save_path = '/home/aoq234/akata-shared/aoq234/mnt/activitynet_features_finetuned_clip_wavcaps'+str(args.index)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model_path = "ViT-B/32"
    save_path = '/home/aoq234/akata-shared/aoq234/mnt/activitynet_features_original_clip_wavcaps'+str(args.index)


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


path=Path("/home/aoq234/akata-shared/datasets/ActivityNet/v1-2-trim") # path to search for videos


dict_csv={}
list_classes=[]
count=0
dict_classes_ids={}
list_of_files=[]

for f in tqdm(path.glob("**/*.mp4")):
    class_name=str(f).split("/")[-2]
    list_of_files.append(f)
    if class_name not in list_classes:
        list_classes.append(class_name)

path=Path("/home/aoq234/akata-shared/datasets/ActivityNet/v1-3-trim")

for f in tqdm(path.glob("**/*.mp4")):
    class_name=str(f).split("/")[-2]
    list_of_files.append(f)
    if class_name not in list_classes:
        list_classes.append(class_name)




chunk=int(len(list_of_files)/3)+1

list_of_files=list_of_files[args.index*chunk:(args.index+1)*chunk]


list_classes.sort()

for index,val in enumerate(sorted(list_classes)):
    dict_classes_ids[val]=index



with open("/home/aoq234/dev/CLIP-GZSL/WavCaps/retrieval/settings/inference.yaml", "r") as f:
    config = yaml.safe_load(f)



wavcaps_model = ASE(config)
wavcaps_model.to(device)

if args.finetuned_model == True:
    cp_path = '/home/aoq234/dev/ClipClap/logs/activitynet_wavcaps_finetuning/epoch20_lr1e6_activity_Sep18_08-11-22_070278_callisto/checkpoints/WavCaps_ActivityNet_finetuned.pt'
    state_dict_key = 'model_state_dict'
else:
    cp_path = '/home/aoq234/dev/CLIP-GZSL/WavCaps/retrieval/pretrained_models/audio_encoders/HTSAT_BERT_zero_shot.pt'
    state_dict_key = 'model'

cp = torch.load(cp_path)
wavcaps_model.load_state_dict(cp[state_dict_key])
wavcaps_model.eval()
print("Model weights loaded from {}".format(cp_path))



counter=0

for f in tqdm(list_of_files):



    counter+=1

    if counter%1000==0:
        with open(save_path, 'wb') as handle:
            pickle.dump(output_list_no_average, handle, protocol=pickle.HIGHEST_PROTOCOL)


    try:
        # audio
        mp4_version = AudioSegment.from_file(str(f), "mp4")
        mp4_version.export("/home/aoq234/akata-shared/aoq234/mnt/activity_dummy"+str(args.index)+".wav", format="wav")


        audio = read_prepare_audio("/home/aoq234/akata-shared/aoq234/mnt/activity_dummy"+str(args.index)+".wav", device)

        with torch.no_grad():
            audio_emb = wavcaps_model.encode_audio(audio).squeeze()
        audio_emb = audio_emb.cpu().detach().numpy() # (1024,)

        # get mid index
        cap = cv2.VideoCapture(str(f))
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fc = 0
        ret = True
        while (fc < frameCount and ret):
            ret, image=cap.read()
            if ret==True:
                fc += 1
        mid_frame_idx = (fc // 2)
        cap.release()



        cap = cv2.VideoCapture(str(f))
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video = torch.zeros((1, 3, 224, 224), dtype=torch.float32)

        fc = 0
        ret = True
        p= torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage()
        ])

        while (fc != mid_frame_idx and ret):

            ret, image=cap.read()
            if ret==True:
                fc += 1

                if fc == mid_frame_idx:

                    torch_image=torch.from_numpy(image)
                    torch_image=torch_image.permute(2,0,1)
                    torch_image=p(torch_image)
                    torch_image=preprocess(torch_image) # returns (3, 224, 224)
                    video[0]=torch_image



        cap.release()

        list_clips=[]


        frame = video[0]
        frame = frame.unsqueeze(0).to(device)  # returns shape  (1, 3, 224, 224) = (bs, channels, height, width)


        if fc == 0:
            image_features = np.array([]) # for some videos in activity net, read() returns false on the first frame and fc stays 0. In this case, tcaf repo returns an empty list for temporal features and nan for averaged features
        else:
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











# nohup python clip_feature_extraction/get_clip_features_activitynet.py --finetuned_model False --index 0 --gpu 3 > logs/get_clip_original_features_activitynet_0.log &

# nohup python clip_feature_extraction/get_clip_features_activitynet.py --finetuned_model False --index 1 --gpu 4 > logs/get_clip_original_features_activitynet_1.log &

# nohup python clip_feature_extraction/get_clip_features_activitynet.py --finetuned_model False --index 2 --gpu 5 > logs/get_clip_original_features_activitynet_2.log &
