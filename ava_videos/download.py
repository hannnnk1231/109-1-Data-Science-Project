import os
import wget
import pandas as pd
from os import listdir
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

train_df = pd.read_csv('ava_train_v2.2.csv')
val_df = pd.read_csv('ava_val_v2.2.csv')
train_vname = train_df['id'].unique()
val_vname = val_df['id'].unique()

URL = 'https://s3.amazonaws.com/ava-dataset/trainval/'

with open('ava_file_names_trainval_v2.1.txt','r') as file:
    video_names = file.readlines()

video_names = [vn.rstrip('\n') for vn in video_names]

if not os.path.exists('./train'):
    os.mkdir('./train')
if not os.path.exists('./val'):
    os.mkdir('./val')
if not os.path.exists('./train_edit'):
    os.mkdir('./train_edit')
if not os.path.exists('./val_edit'):
    os.mkdir('./val_edit')

for v in video_names:
    print("\nNow Processing: ", v)
    vname = v.split('.')[0]
    #wget.download(URL+v, v)
    if vname in train_vname:
        wget.download(URL+v, './train/'+v)
        t1 = min(train_df[train_df['id']==vname]['1'])
        t2 = max(train_df[train_df['id']==vname]['1'])
        ffmpeg_extract_subclip('./train/'+v, t1, t2, targetname='./train_edit/'+v)

    elif vname in val_vname:
        wget.download(URL+v, './val/'+v)
        t1 = min(val_df[val_df['id']==vname]['1'])
        t2 = max(val_df[val_df['id']==vname]['1'])
        ffmpeg_extract_subclip('./val/'+v, t1, t2, targetname='./val_edit/'+v)

print("DOWNLOAD COMPLETED")