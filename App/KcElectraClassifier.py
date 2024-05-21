# kc_electra_classifier.py
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split

datapath = "./App/initdata/"

class KcElectraClassifierModel:
    def __init__(self, model_name='beomi/KcELECTRA-base', model_file=datapath + "kc_bert_emotion_classifier.pth",
                 data_file=datapath + 'data_v3.csv', sample_file=datapath + 'sample-demo.txt', max_length=128):
        self.loaded_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=11)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=11)
        self.data_file = data_file
        self.sample_file = sample_file
        self.max_length = max_length
        self.best_loss = float('inf')
        self.no_improve_count = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_save_path = model_file
        self.label_mapping = {'부정': 0, '성공': 1, '포부': 2}
        self.class_name = {1: '성공', 2: '포부', 0: '부정'}

    def load_model(self):
        loaded_model = self.model
        loaded_model.load_state_dict(torch.load(self.model_save_path, map_location=self.device), strict=False)
        self.loaded_model = loaded_model
        loaded_model.eval()

    def prepare_sample(self, filename):
        inList = []
        with open(filename, "r", encoding="utf-8") as inFp:
            inList = inFp.read()
        return inList

    def prepare_target(self, text, verbose=False):
        inList = text.split('\n')
        outList = []

        for inStr in inList:
            if inStr == '\n':
                continue
            inStr = inStr.replace("<p>", "").replace("</p>", "")
            if len(inStr) == 0:
                continue
            tList = inStr.split('.')
            if tList[-1] == '':
                tList = tList[:-1]
            for tStr in tList:
                t = tStr.strip() + '.'
                outList.append(t)

        if verbose:
            for s in outList:
                print(s)

        return outList

    def classify(self, text):
        loaded_model = self.loaded_model
        loaded_model.load_state_dict(torch.load(self.model_save_path, map_location=self.device), strict=False)
        loaded_model.eval()
        inList = self.prepare_target(text)
        input_encodings = self.tokenizer(inList, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            output = loaded_model(**input_encodings)
        logits = output.logits
        predicted_indices = logits.argmax(dim=1)
        outList = []
        outLogits = []
        predicted_label = []
        self.outText = []
        for i, input_text in enumerate(inList):
            predicted_index = predicted_indices[i].item()
            predicted_logit = logits[:, :3][i].tolist()
            outList.append(input_text)
            outLogits.append(predicted_logit)
            predicted_label.append(predicted_index)
        return outLogits, outList
