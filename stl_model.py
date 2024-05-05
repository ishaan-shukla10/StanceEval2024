import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
from data import emotion_mapping
from ml_collections import ConfigDict
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

cfg = ConfigDict()
cfg.model_name = "CAMeL-Lab/bert-base-arabic-camelbert-msa-sixteenth"
cfg.save_name = "camelbert-msa-sixteenth"
cfg.num_classes = 3
cfg.epochs = 20
cfg.learning_rate = 1e-5
cfg.weight_decay = 1e-6
cfg.batch_size = 32
cfg.use_weighted_sampler = False

class StanceEvalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = cfg.num_classes
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.backbone = AutoModel.from_pretrained(cfg.model_name)
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

    def calculate_f1(self, labels, predictions):
        return classification_report(
            torch.concat(labels, dim=0).cpu(),
            torch.concat(predictions, dim=0).cpu(),
            digits=4,
        )
    
    def calc_f1(self, labels, predictions):
        return f1_score(
            torch.concat(labels, dim=0).cpu(),
            torch.concat(predictions, dim=0).cpu(),
            average="macro",
        )

    def accuracy(self, true, pred):
        true = np.array(true)
        pred = np.array(pred)
        acc = np.sum((true == pred).astype(np.float32)) / len(true)
        return acc * 100

    def fit(self, train_loader, val_loader):
        optim = torch.optim.Adam(
            self.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        best_f1 = 0
        for epoch in range(cfg.epochs):
            train_loss = []
            train_preds = []
            train_labels = []

            val_loss = []
            val_preds = []
            val_labels = []
            self.train()
            print(f"Epoch - {epoch+1}/{cfg.epochs}")
            for batch in tqdm(train_loader):
                batch[0] = self.tokenizer(
                    text=list(batch[0]),
                    return_attention_mask=True,
                    max_length=128,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                text = {k: v.to(self.device) for k, v in batch[0].items()}
                labels = batch[1].to(self.device)
                scores = self(text)
                loss = criterion(scores, labels)
                optim.zero_grad()
                loss.backward()
                optim.step()
                train_loss.append(loss.detach().cpu().numpy())
                train_labels.append(batch[1])
                train_preds.append(scores.argmax(dim=-1))
            print(f"Train loss - {sum(train_loss)/len(train_loss)}")
            train_acc = self.accuracy(
                torch.concat(train_labels, dim=0).cpu(),
                torch.concat(train_preds, dim=0).cpu(),
            )
            train_f1 = self.calc_f1(train_labels, train_preds)
            self.eval()
            with torch.no_grad():
                for batch in tqdm(val_loader):
                    batch[0] = self.tokenizer(
                        text=list(batch[0]),
                        return_attention_mask=True,
                        max_length=128,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )
                    text = {k: v.to(self.device) for k, v in batch[0].items()}
                    labels = batch[1].to(self.device)
                    scores = self(text)
                    loss = criterion(scores, labels)
                    val_loss.append(loss.detach().cpu().numpy())
                    val_labels.append(batch[1])
                    val_preds.append(scores.argmax(dim=-1))
                print(f"Validation loss - {sum(val_loss)/len(val_loss)}")
                val_acc = self.accuracy(
                    torch.concat(val_labels, dim=0).cpu(),
                    torch.concat(val_preds, dim=0).cpu(),
                )
                val_f1 = self.calc_f1(val_labels, val_preds)
            print(self.calculate_f1(train_labels, train_preds))
            print(f"Training Accuracy - {train_acc}")
            print(f"Training F1 - {train_f1}")
            print(self.calculate_f1(val_labels, val_preds))
            print(f"Validation Accuracy - {val_acc}")
            print(f"Validation F1 - {val_f1}")
            if train_f1 > best_f1:
                best_f1 = train_f1
                print("Saved")
                torch.save(
                    self.state_dict(),
                    f"./{cfg.save_name}-{str(cfg.use_weighted_sampler)}-{epoch}.pt",
                )

    def test(self, loader):
        criterion = nn.CrossEntropyLoss()
        test_labels = []
        test_preds = []
        test_loss = []
        self.eval()
        with torch.no_grad():
            for batch in tqdm(loader):
                batch[0] = self.tokenizer(
                        text=list(batch[0]),
                        return_attention_mask=True,
                        max_length=128,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )
                text = {k: v.to(self.device) for k, v in batch[0].items()}
                labels = torch.tensor(batch[1]).to(self.device)
                scores = self(text)
                loss = criterion(scores, labels)
                test_loss.append(loss.detach().cpu().numpy())
                test_labels.append(batch[1])
                test_preds.append(scores.argmax(dim=-1))
            print(f"Test loss - {sum(test_loss)/len(test_loss)}")
            print(self.calculate_f1(test_labels,test_preds))
