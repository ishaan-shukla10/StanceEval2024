import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score
from ml_collections import ConfigDict
from torch.utils.data import DataLoader, Dataset

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

cfg = ConfigDict()
cfg.model_name = "UBC-NLP/MARBERT"
emotion_mapping = {
    "Against": 0,
    "None": 1,
    "Favor": 2,
}

class Dataset:
    def __init__(self, test_data_path):
        super().__init__()
        self.dataframe = pd.read_csv(test_data_path)
        self.label_mappings = emotion_mapping
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        content = self.dataframe["text"].to_list()
        return content[index]
        
    def create_loader(self, batch_size):
        return DataLoader(self, batch_size=batch_size, shuffle=False)

class StanceEvalNet(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.linear_layer = nn.Linear(768, self.num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.push_to_device()

    def forward(self, batch):
        x = self.backbone(**batch).pooler_output
        x = self.linear_layer(x)
        return x

    def push_to_device(self):
        self.backbone.to(self.device)
        self.linear_layer.to(self.device)

    def predict(self, test_loader):
        predictions = []
        self.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader):
                inputs = self.tokenizer(
                   text=list(batch),
                   return_attention_mask=True,
                   max_length=128,
                   padding="max_length",
                   truncation=True,
                   return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                scores = self(inputs)
                predictions.extend(scores.argmax(dim=-1).cpu().numpy())
        return predictions 

test_data_path = "/kaggle/input/correct-stance-eval-data-split/StanceEval_test.csv"

test_dataset = Dataset(test_data_path)
test_loader = test_dataset.create_loader(batch_size=32)

model = StanceEvalNet(model_name=cfg.model_name, num_classes=3)
model.load_state_dict(torch.load("/kaggle/input/top3-final-stance-eval-models/marbertv1-False-11.pt"), strict=False)  # Replace with the path to your model

emotion_mapping_inv = {v: k for k, v in emotion_mapping.items()}
predicted_labels = model.predict(test_loader)

predicted_emotions = [emotion_mapping_inv[label] for label in predicted_labels]

predictions_df = pd.DataFrame(predicted_emotions)
predictions_df.to_csv("marbert_test_final.csv", index=False)
