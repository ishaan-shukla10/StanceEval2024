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
cfg.model_name = "UBC-NLP/MARBERT"
cfg.save_name = "marbertv1"
cfg.num_classes = 3
cfg.epochs = 20
cfg.learning_rate = 1e-5
cfg.weight_decay = 1e-6
cfg.batch_size = 20
cfg.use_weighted_sampler = False

class StanceEvalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_sentiment_classes = 3
        self.num_stance_classes = 3
        self.num_sarcasm_classes = 2
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.backbone = AutoModel.from_pretrained(cfg.model_name)
        self.linear_layer_sentiment = nn.Linear(768, self.num_sentiment_classes)
        self.linear_layer_stance = nn.Linear(768, self.num_stance_classes)
        self.linear_layer_sarcasm = nn.Linear(768, self.num_sarcasm_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.push_to_device()

    def forward(self, batch):
        x = self.backbone(**batch).pooler_output
        sentiment_output = self.linear_layer_sentiment(x)
        stance_output = self.linear_layer_stance(x)
        sarcasm_output = self.linear_layer_sarcasm(x)
        
        return sentiment_output, stance_output, sarcasm_output

    def push_to_device(self):
        self.backbone.to(self.device)
        self.linear_layer_sentiment.to(self.device)
        self.linear_layer_stance.to(self.device)
        self.linear_layer_sarcasm.to(self.device)

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
            train_preds_sentiment = []
            train_preds_stance = []
            train_preds_sarcasm = []
            train_labels_sentiment = []
            train_labels_stance = []
            train_labels_sarcasm = []

            val_loss = []
            val_preds_sentiment = []
            val_preds_stance = []
            val_preds_sarcasm = []
            val_labels_sentiment = []
            val_labels_stance = []
            val_labels_sarcasm = []
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
                labels_sentiment, labels_stance, labels_sarcasm = batch[1], batch[2], batch[3]
                labels_sentiment = labels_sentiment.to(self.device)
                labels_stance = labels_stance.to(self.device)
                labels_sarcasm = labels_sarcasm.to(self.device)
                scores_sentiment, scores_stance, scores_sarcasm = self(text)
                loss_sentiment = criterion(scores_sentiment, labels_sentiment)
                loss_stance = criterion(scores_stance, labels_stance)
                loss_sarcasm = criterion(scores_sarcasm, labels_sarcasm)
                loss = 0.1 * loss_sentiment + 0.4 * loss_stance + 0.1 * loss_sarcasm
                optim.zero_grad()
                loss.backward()
                optim.step()
                train_loss.append(loss.detach().cpu().numpy())
                train_labels_sentiment.append(labels_sentiment)
                train_labels_stance.append(labels_stance)
                train_labels_sarcasm.append(labels_sarcasm)
                train_preds_sentiment.append(scores_sentiment.argmax(dim=-1))
                train_preds_stance.append(scores_stance.argmax(dim=-1))
                train_preds_sarcasm.append(scores_sarcasm.argmax(dim=-1))
            print(f"Train loss - {sum(train_loss)/len(train_loss)}")
            train_acc_sentiment = self.accuracy(
                torch.concat(train_labels_sentiment, dim=0).cpu(),
                torch.concat(train_preds_sentiment, dim=0).cpu(),
            )
            train_acc_stance = self.accuracy(
                torch.concat(train_labels_stance, dim=0).cpu(),
                torch.concat(train_preds_stance, dim=0).cpu(),
            )
            train_acc_sarcasm = self.accuracy(
                torch.concat(train_labels_sarcasm, dim=0).cpu(),
                torch.concat(train_preds_sarcasm, dim=0).cpu(),
            )
            train_f1_sentiment = self.calc_f1(train_labels_sentiment, train_preds_sentiment)
            train_f1_stance = self.calc_f1(train_labels_stance, train_preds_stance)
            train_f1_sarcasm = self.calc_f1(train_labels_sarcasm, train_preds_sarcasm)
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
                    labels_sentiment, labels_stance, labels_sarcasm = batch[1], batch[2], batch[3]
                    labels_sentiment = labels_sentiment.to(self.device)
                    labels_stance = labels_stance.to(self.device)
                    labels_sarcasm = labels_sarcasm.to(self.device)
                    scores_sentiment, scores_stance, scores_sarcasm = self(text)
                    loss_sentiment = criterion(scores_sentiment, labels_sentiment)
                    loss_stance = criterion(scores_stance, labels_stance)
                    loss_sarcasm = criterion(scores_sarcasm, labels_sarcasm)
                    loss = 0.1 * loss_sentiment + 0.4 * loss_stance + 0.1 * loss_sarcasm
                    val_loss.append(loss.detach().cpu().numpy())
                    val_labels_sentiment.append(labels_sentiment)
                    val_labels_stance.append(labels_stance)
                    val_labels_sarcasm.append(labels_sarcasm)
                    val_preds_sentiment.append(scores_sentiment.argmax(dim=-1))
                    val_preds_stance.append(scores_stance.argmax(dim=-1))
                    val_preds_sarcasm.append(scores_sarcasm.argmax(dim=-1))
                print(f"Validation loss - {sum(val_loss)/len(val_loss)}")
                val_acc_sentiment = self.accuracy(
                    torch.concat(val_labels_sentiment, dim=0).cpu(),
                    torch.concat(val_preds_sentiment, dim=0).cpu(),
                )
                val_acc_stance = self.accuracy(
                    torch.concat(val_labels_stance, dim=0).cpu(),
                    torch.concat(val_preds_stance, dim=0).cpu(),
                )
                val_acc_sarcasm = self.accuracy(
                    torch.concat(val_labels_sarcasm, dim=0).cpu(),
                    torch.concat(val_preds_sarcasm, dim=0).cpu(),
                )
                val_f1_sentiment = self.calc_f1(val_labels_sentiment, val_preds_sentiment)
                val_f1_stance = self.calc_f1(val_labels_stance, val_preds_stance)
                val_f1_sarcasm = self.calc_f1(val_labels_sarcasm, val_preds_sarcasm)
            print(self.calculate_f1(train_labels_sentiment, train_preds_sentiment))
            print(self.calculate_f1(train_labels_stance, train_preds_stance))
            print(self.calculate_f1(train_labels_sarcasm, train_preds_sarcasm))
            print(f"Training Accuracy (Sentiment)- {train_acc_sentiment}")
            print(f"Training Accuracy (Stance)- {train_acc_stance}")
            print(f"Training Accuracy (Sarcasm)- {train_acc_sarcasm}")
            print(f"Training F1 (Sentiment)- {train_f1_sentiment}")
            print(f"Training F1 (Stance)- {train_f1_stance}")
            print(f"Training F1 (Sarcasm)- {train_f1_sarcasm}")
            print(self.calculate_f1(val_labels_sentiment, val_preds_sentiment))
            print(self.calculate_f1(val_labels_stance, val_preds_stance))
            print(self.calculate_f1(val_labels_sarcasm, val_preds_sarcasm))
            print(f"Validation Accuracy (Sentiment)- {val_acc_sentiment}")
            print(f"Validation Accuracy (Stance)- {val_acc_stance}")
            print(f"Validation Accuracy (Sarcasm)- {val_acc_sarcasm}")
            print(f"Validation F1 (Sentiment)- {val_f1_sentiment}")
            print(f"Validation F1 (Stance)- {val_f1_stance}")
            print(f"Validation F1 (Sarcasm)- {val_f1_sarcasm}")
            if train_f1_stance > best_f1:
                best_f1 = train_f1_stance
                print("Saved")
                torch.save(
                    self.state_dict(),
                    f"./{cfg.save_name}-{str(cfg.use_weighted_sampler)}-{epoch}.pt",
                )

    def test(self, loader):
        criterion = nn.CrossEntropyLoss()
        test_labels_sentiment = []
        test_labels_stance = []
        test_labels_sarcasm = []
        test_preds_sentiment = []
        test_preds_stance = []
        test_preds_sarcasm = []
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
                labels_sentiment, labels_stance, labels_sarcasm = batch[1], batch[2], batch[3]
                labels_sentiment = labels_sentiment.to(self.device)
                labels_stance = labels_stance.to(self.device)
                labels_sarcasm = labels_sarcasm.to(self.device)
                scores_sentiment, scores_stance, scores_sarcasm = self(text)
                loss_sentiment = criterion(scores_sentiment, labels_sentiment)
                loss_stance = criterion(scores_stance, labels_stance)
                loss_sarcasm = criterion(scores_sarcasm, labels_sarcasm)
                loss = 0.1 * loss_sentiment + 0.4 * loss_stance + 0.1 * loss_sarcasm
                test_loss.append(loss.detach().cpu().numpy())
                test_labels_sentiment.append(labels_sentiment)
                test_labels_stance.append(labels_stance)
                test_labels_sarcasm.append(labels_sarcasm)
                test_preds_sentiment.append(scores_sentiment.argmax(dim=-1))
                test_preds_stance.append(scores_stance.argmax(dim=-1))
                test_preds_sarcasm.append(scores_sarcasm.argmax(dim=-1))
            print(f"Test loss - {sum(test_loss)/len(test_loss)}")
            print(self.calculate_f1(test_labels_sentiment, test_preds_sentiment))
            print(self.calculate_f1(test_labels_stance, test_preds_stance))
            print(self.calculate_f1(test_labels_sarcasm, test_preds_sarcasm))
