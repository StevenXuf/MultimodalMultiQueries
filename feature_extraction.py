import torch
import fire
from tqdm import tqdm
import os
import sys
import glob
import librosa
import librosa.display
import simplejpeg

import numpy as np
import torchvision as tv
import matplotlib.pyplot as plt
import torch.nn.functional as F

from PIL import Image
from torch.nn.utils.rnn import pad_sequence 
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from torchmetrics.retrieval import RetrievalRecall

from get_datasets import get_all_loaders,get_default_config

sys.path.append(os.path.abspath("AudioCLIP"))
from model import AudioCLIP

def get_targets(va_dist_art2audio,k):
    _,index = torch.topk(va_dist_art2audio,k=k,dim=1)
    targets = torch.zeros(*va_dist_art2audio.size(), dtype=torch.long)
    ones = torch.ones_like(index, dtype=torch.long)

    return targets.scatter_(dim=1, index=index, src=ones)

def get_metrics(sim,va_distance,k,device):     
    compute_recall=RetrievalRecall(top_k=k)

    targets=get_targets(-va_distance,k).to(device)
                                              
    indexes = torch.arange(sim.size(0), dtype=torch.long).unsqueeze(1).expand(*sim.size()).to(device)
    recall=compute_recall(sim,targets,indexes=indexes)
    return recall

    
def extract_features(model,config_path='./config.yaml',SEQ_LENGTH=None,TOP_K=None):
    device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    cfg=get_default_config(config_path)
    if SEQ_LENGTH is None:
        SEQ_LENGTH=cfg['AudioCLIP']['SEQ_LENGTH']
    if TOP_K is None:
        TOP_K=cfg['General']['TOP_K']

    info=get_all_loaders()
    wikiart_loader=info['wikiart']
    deam_train_loader=info['deam_train']
    deam_test_loader=info['deam_test']
    va_dist_art2audio=info['va_dist_art2audio_desc']
    
    model.to(device)

    with torch.no_grad():
        audio_features=[]
        for train_batch in tqdm(deam_train_loader):
            padded_batch = pad_sequence(train_batch['audio'], batch_first=True, padding_value=0.0)[:,SEQ_LENGTH]
            padded_batch=padded_batch.unsqueeze(1).to(device)
            ((audio_embeddings, _, _), _), _ = model(audio=padded_batch)
            audio_features.append(audio_embeddings)
        
        audio_features=F.normalize(torch.cat(audio_features,dim=0),p=2,dim=1)

        image_features,text_features=[],[]
        for wiki_batch in tqdm(wikiart_loader):
            categories = wiki_batch['category']
            images = wiki_batch["images"].to(device)        # Tensor shape: [B, 3, H, W]
            titles = wiki_batch["titles"]        # List of titles
            years = wiki_batch["years"]          # List of years
            artists = wiki_batch["artists"]      # List of artists
            texts=[[f'The painting {title} is, a {category}, created by {artist} in {year}'] for (title,category,artist,year) in zip(titles,categories,artists,years)]
            ((_, image_embeddings, _), _), _ = model(image=images)
            image_features.append(image_embeddings)

            ((_, _, text_embeddings), _), _ = model(text=texts)
            text_features.append(text_embeddings)
        
        image_features=F.normalize(torch.cat(image_features,dim=0),p=2,dim=1)
        text_features=F.normalize(torch.cat(text_features,dim=0),p=2,dim=1)
    
    i2a_sim=pairwise_cosine_similarity(image_features,audio_features)
    t2a_sim=pairwise_cosine_similarity(text_features,audio_features)
    
    recall_i2a=get_metrics(i2a_sim,va_dist_art2audio,TOP_K,device)
    recall_t2a=get_metrics(t2a_sim,va_dist_art2audio,TOP_K,device)
    
    print(f'Recall@{TOP_K}: image-to-audio {recall_i2a:.2f}: text-to-audio {recall_t2a:.2f}')
    
    #image+text---->audio
    added_features=F.normalize(image_features+text_features,p=2,dim=1)
    multiplied_features=F.normalize(image_features*text_features,p=2,dim=1)
    
    added_sim=pairwise_cosine_similarity(added_features,audio_features)
    multiplied_sim=pairwise_cosine_similarity(multiplied_features,audio_features)

    recall_added=get_metrics(added_sim,va_dist_art2audio,TOP_K,device)
    recall_multiplied=get_metrics(multiplied_sim,va_dist_art2audio,TOP_K,device)

    print(f'Recall@{TOP_K}: add(image,text)-to-audio {recall_added:.2f}: multiply(image,text)-to-audio {recall_multiplied:.2f}')
    
if __name__=='__main__':
    torch.manual_seed(0)

    MODEL_FILENAME = 'AudioCLIP-Full-Training.pt'
    model = AudioCLIP(pretrained=f'./AudioCLIP/assets/{MODEL_FILENAME}')
    extract_features(model)
