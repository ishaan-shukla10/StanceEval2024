import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from ml_collections import ConfigDict

emotion_mapping = {
    "Against": 0,
    "None": 1,
    "Favor": 2,
}

cfg = ConfigDict()
cfg.batch_size = 32

class Dataset:
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        if self.mode == "train":
            self.dataframe = entire_df
        elif self.mode == "test":
            self.dataframe = test_df
        elif self.mode == "val":
            self.dataframe = val_df
        self.label_mappings = emotion_mapping
        self.dataframe["stance"] = self.dataframe["stance"].replace(np.nan, "None")
        self.dataframe["labels"] = self.dataframe["stance"].map(self.label_mappings)
        self.dataframe["labels"] = self.dataframe["labels"].astype(int)
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        content = self.dataframe["text"].to_list()
        sentiment = self.dataframe["labels"].to_list()
        return content[index], sentiment[index]
        
    def create_loader(self):
        if self.mode == "train":
            return DataLoader(self, batch_size=cfg.batch_size, shuffle=True,num_workers = 2)
        elif self.mode == "val" or self.mode == "test":
            return DataLoader(self, batch_size=cfg.batch_size, shuffle=False,num_workers = 2)
                
