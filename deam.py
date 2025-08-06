import os
import torch
import torchaudio
import pandas as pd

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch.nn.utils.rnn import pad_sequence

from configuration import get_default_config


class AudioDataset(Dataset):
    def __init__(self, folder_path, df_info, transform=None, target_sample_rate=44100):
        self.folder_path = folder_path
        self.df_info = df_info
        self.song_ids = [str(song_id)+'.mp3' for song_id in self.df_info['song_id'].tolist()]  # Assuming 'song_id' is the column with audio file names
        self.files = [f for f in os.listdir(folder_path) if f.endswith('.mp3') and f in self.song_ids]
        self.transform = transform
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        song_id = self.files[idx].split('.')[0]  # Extract song_id from filename
        valence_mean,arousal_mean = self.df_info.loc[self.df_info['song_id'] == int(song_id), ['valence_mean', 'arousal_mean']].iloc[0]
        file_path = os.path.join(self.folder_path, self.files[idx])
        
        # Load audio
        waveform, sample_rate = torchaudio.load(file_path)  # shape: (channels, samples)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate,new_freq=self.target_sample_rate)
            waveform = resampler(waveform)

        waveform = waveform/waveform.abs().max()  # Normalize waveform
        
        # Apply any other transform (e.g., spectrogram)
        if self.transform:
            waveform = self.transform(waveform)
        
        return waveform, valence_mean, arousal_mean  # returning filename as label for now

def collate_fn(batch):
    waveforms, valence_mean, arousal_mean = zip(*batch)
    # Remove channel dimension for pad_sequence (if mono)
    waveforms = [w.squeeze(0) for w in waveforms]
    padded_waveforms = pad_sequence(waveforms, batch_first=True)[:,:441000]  # Limit to 10 second of audio
    valence_mean = torch.tensor(valence_mean, dtype=torch.float32)
    arousal_mean = torch.tensor(arousal_mean, dtype=torch.float32)
    return {'audio': padded_waveforms, 
            'valence_mean': valence_mean,
            'arousal_mean': arousal_mean}


def get_deam_loader(config_path='./config.yaml'):
    cfg=get_default_config(config_path)
    audio_path=cfg['DEAM']['AUDIO_PATH']
    annotation_path=cfg['DEAM']['ANNOTATION_PATH']
    sample_rate=cfg['DEAM']['SAMPLING_RATE']
    threshold_valence=cfg['DEAM']['THRESHOLD_VALENCE']
    threshold_arousal=cfg['DEAM']['THRESHOLD_AROUSAL']

    df1=pd.read_csv(os.path.join(annotation_path,'static_annotations_averaged_songs_1_2000.csv'),header=0,sep=',')
    df1.columns = df1.columns.str.strip()
    df2=pd.read_csv(os.path.join(annotation_path,'static_annotations_averaged_songs_2000_2058.csv'),header=0,sep=',')
    df2.columns = df2.columns.str.strip()
    df_combined = pd.concat([df1, df2], ignore_index=True)
    print(df_combined.shape)
    filtered_df = df_combined[
    (df_combined['valence_std'] < threshold_valence) &
    (df_combined['arousal_std'] < threshold_arousal)
    ]
    print(f"Filtered dataset size: {filtered_df.shape}")

    dataset = AudioDataset(folder_path=audio_path, df_info=filtered_df,target_sample_rate=sample_rate)
    print(f"Dataset size: {len(dataset)}")
    deam_loader = DataLoader(dataset, batch_size=cfg['General']['BATCH_SIZE'],
                             shuffle=False, num_workers=cfg['General']['NUM_WORKERS'],
                             collate_fn=collate_fn)

    return deam_loader

if __name__ == "__main__":
    dataloader=get_deam_loader()
    for batch in dataloader:
        padded_waveforms, valence_mean, arousal_mean = batch['audio'], batch['valence_mean'], batch['arousal_mean']
        print(f"Padded waveforms shape: {padded_waveforms.shape}, valence shape: {valence_mean.shape}, arousal shape: {arousal_mean.shape}")