import torch
import fire

from get_datasets import get_all_loaders

def build_triplets():
    wikiart_loader,deam_train_loader,deam_test_loader=get_all_loaders()
    
    wiki_va=[]
    for wiki_batch in wikiart_loader:
        category = wiki_batch['category']
        images = wiki_batch["images"]        # Tensor shape: [B, 3, H, W]
        titles = wiki_batch["titles"]        # List of titles
        years = wiki_batch["years"]          # List of years
        artists = wiki_batch["artists"]      # List of artists
        wiki_valence=wiki_batch['valence']
        wiki_arousal=wiki_batch['arousal']
        wiki_va.append(torch.cat([wiki_valence,wiki_arousal],dim=1))
    
    deam_train_va=[]
    for train_batch in deam_train_loader:
        deam_train_valence=train_batch['valence_mean']
        deam_train_arousal=train_batch['arousal_mean']
        #valence_distance=torch.norm(wiki_valence-deam_train_valence,p=2)
        #arousal_distance=torch.norm(wiki_arousal-deam_train_arousal,p=2)
        deam_train_va.append(torch.cat([train_batch['valence_mean'].unsqueeze(1),train_batch['arousal_mean'].unsqueeze(1)],dim=1))

    wiki_va=torch.cat(wiki_va,dim=0)
    deam_train_va=torch.cat(deam_train_va,dim=0)

    print(wiki_va.size())

if __name__=='__main__':
    build_triplets()
