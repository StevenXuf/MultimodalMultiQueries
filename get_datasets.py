import torch
import pandas as pd
import requests
import yaml
import fire

from datasets import load_dataset
from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import to_tensor
from torchmetrics.functional.pairwise import pairwise_euclidean_distance

from text_to_vad import get_vad,get_lexicon

def get_deam_loaders(threshold_valence=1.5,threshold_arousal=1.5,batch_size=64):
    ds = load_dataset("Rehead/DEAM_stripped_vocals")
    train,test=ds['train'],ds['test']
    filtered_train=filter_dataset(train,threshold_valence=threshold_valence,threshold_arousal=threshold_arousal)
    filtered_test=filter_dataset(test,threshold_valence=threshold_valence,threshold_arousal=threshold_arousal)
    return get_dataloader(filtered_train,deam_collate_fn,batch_size),get_dataloader(filtered_test,deam_collate_fn,batch_size)

def filter_dataset(dataset,threshold_valence=1.5,threshold_arousal=1.5):
    # Assuming `dataset` is your Hugging Face Dataset object
    # Filter the dataset
    filtered_dataset = dataset.filter(
        lambda example: example['valence_std'] < threshold_valence and example['arousal_std'] < threshold_arousal)

    # Set format for non-audio columns to PyTorch tensors
    non_audio_cols = [col for col in filtered_dataset.column_names if col != 'audio']
    filtered_dataset.set_format(type='torch', columns=non_audio_cols, output_all_columns=True)
    
    def normalize(example):
        valence_min = filtered_dataset["valence_mean"].min()
        valence_max = filtered_dataset["valence_mean"].max()
        arousal_min = filtered_dataset["arousal_mean"].min()
        arousal_max = filtered_dataset["arousal_mean"].max()
        example["valence_mean"] = (example["valence_mean"] - valence_min) / (valence_max - valence_min)
        example["arousal_mean"] = (example["arousal_mean"] - arousal_min) / (arousal_max - arousal_min)
        return example
    filtered_dataset=filtered_dataset.map(normalize)

    return filtered_dataset
    

    # Custom collate function to process audio
def deam_collate_fn(batch):
    audio_data = [item.pop('audio') for item in batch]  # Extract audio
    collated_batch = default_collate(batch)  # Collate non-audio features

    # Convert audio arrays to tensors and add to batch
    audio_arrays = [torch.as_tensor(audio['array']) for audio in audio_data]
    collated_batch['audio'] = audio_arrays

    return collated_batch

def get_dataloader(dataset,collate_fn,batch_size=64):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,  # Adjust batch size as needed
        shuffle=True,    # Optional: Shuffle for training
        collate_fn=collate_fn
    )
    return dataloader

def get_img_transform(IMAGE_SIZE=224):
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.Lambda(lambda img: to_tensor(img)),  # Convert to tensor [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform


class ArtworkDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image from URL
        try:
            response = requests.get(row['Image URL'], timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as e:
            print(f"Error loading image for ID {row['ID']}: {e}")
            # Return placeholder image on failure
            img = Image.new('RGB', (IMAGE_SIZE,IMAGE_SIZE), (128, 128, 128))

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        # Process metadata
        category = str(row['Category']) if pd.notna(row['Category']) else "Unknown"
        title = str(row['Title']) if pd.notna(row['Title']) else ""
        year = str(row['Year']) if pd.notna(row['Year']) else "Unknown"
        artist = str(row['Artist']) if pd.notna(row['Artist']) else "Unknown"
        valence=torch.tensor([row['valence']])
        arousal=torch.tensor([row['arousal']])

        return category, img, title, year, artist, valence, arousal

# Custom collate function to handle text metadata
def custom_art_collate(batch):
    category, images, titles, years, artists, valence, arousal= zip(*batch)

    # Stack images into tensor
    images = torch.stack(images)
    valence=torch.stack(valence)
    arousal=torch.stack(arousal)

    # Return metadata as lists
    return {
        "category": category,
        "images": images,
        "titles": titles,
        "years": years,
        "artists": artists,
        "valence": valence,
        "arousal": arousal
    }

# Load and prepare data
def create_wikiart_dataloader(df,custom_art_collate,BATCH_SIZE=64,IMAGE_SIZE=224):
    # Create dataset and dataloader
    dataset = ArtworkDataset(df, transform=get_img_transform(IMAGE_SIZE))

    return get_dataloader(dataset,custom_art_collate,BATCH_SIZE)

def get_image_title_emotion(emotion_file,lexicon):
    df=pd.read_csv(emotion_file,sep='\t')
    #cols_to_keep=[col for col in df.columns if 'Art (image+title)' in col]
    #df=df[cols_to_keep] 
    df['valence'],df['arousal']=None,None

    for idx, row in df.iterrows():
        emotions=[item.split(':')[1].strip() for item in list(row[9:69].index[row[9:69]== 1])]
        va=get_vad(' '.join(emotions),lexicon)
        df.at[idx,'valence'],df.at[idx,'arousal']=va['valence'],va['arousal']
    
    return df

def get_shared_info(df1,df2):
    df_common = pd.merge(df1,df2,on=['ID', 'Category', 'Artist', 'Title', 'Year'],how='inner')
    return df_common

def get_default_config(config_path):
    with open(config_path,'r') as f:
        return yaml.safe_load(f)



def get_all_loaders(config='./config.yaml',BATCH_SIZE=None,TSV_PATH=None,IMAGE_SIZE=None,EMOTION_PATH=None, LEXICON_PATH=None,THRESHOLD_VALENCE=None,THRESHOLD_AROUSAL=None,TOP_K=None):
    
    cfg=get_default_config('./config.yaml')

    if BATCH_SIZE is None:
        BATCH_SIZE=cfg['General']['BATCH_SIZE']
    if TSV_PATH is None:
        TSV_PATH=cfg['WikiArt']['TSV_PATH']
    if IMAGE_SIZE is None:
        IMAGE_SIZE=cfg['WikiArt']['IMAGE_SIZE']
    if EMOTION_PATH is None:
        EMOTION_PATH=cfg['WikiArt']['EMOTION_PATH']
    if LEXICON_PATH is None:
        LEXICON_PATH=cfg['Lexicon']['LEXICON_PATH']
    if THRESHOLD_VALENCE is None:
        THRESHOLD_VALENCE=cfg['DEAM']['THRESHOLD_VALENCE']
    if THRESHOLD_AROUSAL is None:
        THRESHOLD_AROUSAL=cfg['DEAM']['THRESHOLD_AROUSAL']
    if TOP_K is None:
        TOP_K=cfg['General']['TOP_K']

    lexicon=get_lexicon(LEXICON_PATH)
    df_emo=get_image_title_emotion(EMOTION_PATH,lexicon)
    df_main=pd.read_csv(TSV_PATH,sep='\t')
    df_main=df_main.dropna(subset=['Image URL'])
    df=get_shared_info(df_main,df_emo) 

    wikiart_loader = create_wikiart_dataloader(df,custom_art_collate,BATCH_SIZE,IMAGE_SIZE)

    train_loader,test_loader=get_deam_loaders(THRESHOLD_VALENCE,THRESHOLD_AROUSAL,BATCH_SIZE)
    
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
    for train_batch in train_loader:
        deam_train_valence=train_batch['valence_mean']
        deam_train_arousal=train_batch['arousal_mean']
        deam_train_va.append(torch.cat([train_batch['valence_mean'].unsqueeze(1),train_batch['arousal_mean'].unsqueeze(1)],dim=1))

    wiki_va=torch.cat(wiki_va,dim=0)
    deam_train_va=torch.cat(deam_train_va,dim=0)
    distance=pairwise_euclidean_distance(wiki_va,deam_train_va)
    
    _,i2a_ids=torch.topk(-distance,k=TOP_K,dim=1)
    _,a2i_ids=torch.topk(-distance.T,k=TOP_K,dim=1)

    return {'wikiart':wikiart_loader,
            'deam_train':train_loader,
            'deam_test':test_loader,
            'img2audio_ids':i2a_ids,
            'audio2img_ids':a2i_ids}


if __name__ == "__main__":
    fire.Fire(get_all_loaders)
