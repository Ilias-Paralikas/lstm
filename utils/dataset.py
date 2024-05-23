import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess(df,FILL_NAN):
    # fill nan values
    df_ffill = df.ffill(limit=FILL_NAN)
    df_bfill = df.bfill(limit=FILL_NAN)
    df = (df_ffill + df_bfill) / 2
    # scaler
    scaler = StandardScaler()
    df  = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df


def train_test_split(df,split):
    train_split = int(len(df)*split)
    test_split=  len(df) -train_split
    return train_split,test_split

class LoadPredictionDataset(Dataset):
    def __init__(self,df,
                 start_index,
                 population,                 
                 time_step,
                 column,
                 device):
        previous_overflow=  max(start_index-time_step,0)
        df = df[column]
        self.df = df.iloc[previous_overflow:start_index+population]
        self.column = column
        self.time_step = time_step
        self.device = device
        self.length = len(self.df)-self.time_step -1
        
    def __getitem__(self, index):
        previous_values = self.df.iloc[index:index+self.time_step].values
        previous_values = torch.tensor(previous_values).unsqueeze(0)
        previous_values = previous_values.float().to(self.device)
        target_values = self.df.iloc[index+self.time_step]
        target_values = torch.tensor(target_values).float().to(self.device)
        target_values = target_values.unsqueeze(0)
        return previous_values, target_values
    
    def __len__(self):
        return self.length