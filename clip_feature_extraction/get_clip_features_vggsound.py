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



parser=argparse.ArgumentParser(description="GZSL with ESZSL")
parser.add_argument('--finetuned_model', type=str_to_bool, default=False,
                    help='Whether to load original Clip and WavCaps models or the finetuned verisons')
parser.add_argument('--index', default=0, type=int)
parser.add_argument('--gpu', default=0, type=int)
args=parser.parse_args()

# device = 'cuda:4'
device = 'cuda:'+str(args.gpu)
model, preprocess = clip.load("ViT-B/32", device=device)

if args.finetuned_model == True:
    raise NotImplementedError()

else:
    model_path = "ViT-B/32"
    save_path = '/home/aoq234/akata-shared/aoq234/mnt/vggsound_features_original_clip_wavcaps'+str(args.index)

model = model.to(device)
model.eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)






with open("/home/aoq234/dev/CLIP-GZSL/WavCaps/retrieval/settings/inference.yaml", "r") as f:
    config = yaml.safe_load(f)

wavcaps_model = ASE(config)
wavcaps_model.to(device)

if args.finetuned_model == True:
    raise NotImplementedError()

else:
    cp_path = '/home/aoq234/dev/CLIP-GZSL/WavCaps/retrieval/pretrained_models/audio_encoders/HTSAT_BERT_zero_shot.pt'
    state_dict_key = 'model'

cp = torch.load(cp_path)
wavcaps_model.load_state_dict(cp[state_dict_key])
wavcaps_model.eval()
print("Model weights loaded from {}".format(cp_path))






output_list_no_average=[]


path=Path("/home/aoq234/akata-shared/datasets/vggsound/video")

dict_csv={}
list_classes=[]
count=0
with open('/home/aoq234/akata-shared/datasets/vggsound/metadata/vggsound.csv', 'r') as read_obj: # path of the metadata
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Iterate over each row in the csv using reader object
    for row in csv_reader:
        key=str(row[0])+"_"+str(row[1])
        if row[2] not in list_classes:
            list_classes.append(row[2])
        dict_csv[key]=[row[2],row[3]]

list_classes.sort()
dict_classes_ids={}
for index,val in enumerate(sorted(list_classes)):
    dict_classes_ids[val]=index

list_of_files=[]
for f in tqdm(path.glob("**/*.mp4")):
        list_of_files.append(f)

chunk=int(len(list_of_files)/3)+1

list_of_files=list_of_files[args.index*chunk:(args.index+1)*chunk]


counter=0

for f in tqdm(list_of_files):

    counter+=1
    if counter%3000==0:
        with open(save_path, 'wb') as handle:
            pickle.dump(output_list_no_average, handle, protocol=pickle.HIGHEST_PROTOCOL)


    try:
        # audio
        mp4_version = AudioSegment.from_file(str(f), "mp4")
        mp4_version.export("/home/aoq234/akata-shared/aoq234/mnt/vggsound_dummy"+str(args.index)+".wav", format="wav")


        audio = read_prepare_audio("/home/aoq234/akata-shared/aoq234/mnt/vggsound_dummy"+str(args.index)+".wav", device)

        with torch.no_grad():
            audio_emb = wavcaps_model.encode_audio(audio).squeeze()
        audio_emb = audio_emb.cpu().detach().numpy() # (1024,)


        p= torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage()
        ])

        # get mid index
        while True:
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
            if fc!=0:
                break

        while True:
            cap = cv2.VideoCapture(str(f))
            frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video = torch.zeros((1,  3, 224, 224), dtype=torch.float32)

            fc = 0
            ret = True

            while (fc < frameCount and ret):
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
            if fc!=0:
                break

        list_clips=[]


        frame = video[0]
        frame = frame.unsqueeze(0).to(device)  # returns shape  (1, 3, 224, 224) = (bs, channels, height, width)


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
    splitted_path=name_file.rsplit('_', 1)
    class_id=splitted_path[0]
    number=int(splitted_path[1].split(".")[0])
    search_name=class_id+"_"+str(number)
    class_name=dict_csv[search_name][0]
    class_id=dict_classes_ids[class_name]

    result_list=[image_features, class_id, audio_emb, name_file]


    output_list_no_average.append(result_list)





with open(save_path, 'wb') as handle:
    pickle.dump(output_list_no_average, handle, protocol=pickle.HIGHEST_PROTOCOL)





# nohup python clip_feature_extraction/get_clip_features_vggsound.py --finetuned_model False --index 0 --gpu 1 > logs/get_clip_features_vggsound_0.log &

# nohup python clip_feature_extraction/get_clip_features_vggsound.py --finetuned_model False --index 1 --gpu 2 > logs/get_clip_features_vggsound_1.log &

# nohup python clip_feature_extraction/get_clip_features_vggsound.py --finetuned_model False --index 2 --gpu 3 > logs/get_clip_features_vggsound_2.log &
