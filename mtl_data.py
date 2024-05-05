import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from ml_collections import ConfigDict

sentiment_mapping = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2,
}

stance_mapping = {
    "Against":  0,
    "None": 1,
    "Favor": 2,
}

sarcasm_mapping = {
    "No": 0,
    "Yes": 1,
}

cfg = ConfigDict()
cfg.batch_size = 32

df = pd.read_csv("StanceEval_MTL_processed.csv")

covid_list = []
women_list = []
digital_list = []

for i in range(len(df["text"])):
    if df["target"][i] == "Women empowerment":
        women_list.append([df["text"][i], df["sentiment"][i], df["stance"][i], df["sarcasm"][i]])
    elif df["target"][i] == "Covid Vaccine":
        covid_list.append([df["text"][i], df["sentiment"][i], df["stance"][i], df["sarcasm"][i]])
    else:
        digital_list.append([df["text"][i], df["sentiment"][i], df["stance"][i], df["sarcasm"][i]])

covid_arr = np.array(covid_list)
women_arr = np.array(covid_list)
digital_arr = np.array(digital_list)
from sklearn.model_selection import train_test_split

covid_train, covid_temp = train_test_split(covid_arr, test_size=0.2)
covid_test, covid_val = train_test_split(covid_temp, test_size=0.5)
digital_train, digital_temp = train_test_split(digital_arr, test_size=0.2)
digital_test, digital_val = train_test_split(digital_temp, test_size=0.5)
women_train, women_temp = train_test_split(women_arr, test_size=0.2)
women_test, women_val = train_test_split(women_temp, test_size=0.5)
covid_train_df = pd.DataFrame(covid_train, columns=["text", "sentiment", "stance", "sarcasm"])
women_train_df = pd.DataFrame(women_train, columns=["text", "sentiment", "stance", "sarcasm"])
digital_train_df = pd.DataFrame(digital_train, columns=["text", "sentiment", "stance", "sarcasm"])
covid_test_df = pd.DataFrame(covid_test, columns=["text", "sentiment", "stance", "sarcasm"])
women_test_df = pd.DataFrame(women_test, columns=["text", "sentiment", "stance", "sarcasm"])
digital_test_df = pd.DataFrame(digital_test, columns=["text", "sentiment", "stance", "sarcasm"])
covid_val_df = pd.DataFrame(covid_val, columns=["text", "sentiment", "stance", "sarcasm"])
women_val_df = pd.DataFrame(women_val, columns=["text", "sentiment", "stance", "sarcasm"])
digital_val_df = pd.DataFrame(digital_val, columns=["text", "sentiment", "stance", "sarcasm"])
final_train_df = pd.concat([covid_train_df, women_train_df, digital_train_df])
final_test_df = pd.concat([covid_test_df, women_test_df, digital_test_df])
final_val_df = pd.concat([covid_val_df, women_val_df, digital_val_df])

class Dataset:
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        if self.mode == "train":
            self.dataframe = final_train_df
        elif self.mode == "test":
            self.dataframe = final_test_df
        elif self.mode == "val":
            self.dataframe = final_val_df
        self.sentiment_mappings = sentiment_mapping
        self.stance_mappings = stance_mapping
        self.sarcasm_mappings = sarcasm_mapping
        self.dataframe["stance"] = self.dataframe["stance"].replace("nan", "None")
        self.dataframe["stance"] = self.dataframe["stance"].map(self.stance_mappings)
        #self.dataframe["stance"] = self.dataframe["stance"].astype(int)
        self.dataframe["sarcasm"] = self.dataframe["sarcasm"].map(self.sarcasm_mappings)
        self.dataframe["sentiment"] = self.dataframe["sentiment"].map(self.sentiment_mappings)

    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        content = self.dataframe["text"].iloc[index]
        sentiment = self.dataframe["sentiment"].iloc[index]
        stance = self.dataframe["stance"].iloc[index]
        sarcasm = self.dataframe["sarcasm"].iloc[index]
        return content, sentiment, stance, sarcasm

        
    def create_loader(self):
        if self.mode == "train":
            return DataLoader(self, batch_size=cfg.batch_size, shuffle=True,num_workers = 2)
        elif self.mode == "val" or self.mode == "test":
            return DataLoader(self, batch_size=cfg.batch_size, shuffle=False,num_workers = 2)
                
