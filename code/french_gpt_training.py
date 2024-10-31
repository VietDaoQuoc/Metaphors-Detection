# -*- coding: utf-8 -*-
"""french gpt_training.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1R3ycebFZH830H4tKTnEiFQA4NpV80x7G
"""

# prep for mount
#from google.colab import drive
#drive.mount('/content/drive')

#!python -m pip install opik

#!python -m pip install wandb -Uq
#!python -m pip install ray[tune]
# !python -m pip install sigopt
# !python -m pip install optuna
#!python -m pip install nltk python-Levenshtein
# !python -m pip install rouge-score
#!python -m pip install peft

#!pip install comet_ml

import comet_ml
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoConfig, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader, random_split
import time
import logging
import numpy as np
#import wandb
import random
import math
import os
import pandas as pd
import re
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein as lev
import nltk
#from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
#from rouge_score import rouge_scorer
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from transformers import GPT2Config,  AutoModelForSeq2SeqLM
from peft import get_peft_model, IA3Config, TaskType
from transformers import TrainerCallback

# os.environ["WANDB_API_KEY"] = "a5d8e3b6d2ef7d55f930ab72670aaa64e1a4198d"

# wandb.login() #enter this token: a5d8e3b6d2ef7d55f930ab72670aaa64e1a4198d
# #wandb.init(project="huggingface", entity="teamproject464-universit-t-des-saarlandes-saarland-unive")

# import opik
# #kPnaiIq69hBXz7vVAWlDTtRed
# opik.configure(use_local=False)

comet_ml.login(api_key='kPnaiIq69hBXz7vVAWlDTtRed', project_name='metaphor_detection', workspace='nadias')
experiment = comet_ml.Experiment()

#!ls '/content/drive/MyDrive'

import pandas as pd
import re
# Preprocessing french data

def clean_output(text):
    # Remove commas, parentheses, and square brackets
    return re.sub(r'[\(\)\[\]\']', '', text)

def clean_output_special(text):
    if pd.isna(text):
        return text
    return re.sub(r'[\[\]\']' , '', text)

df_literal = pd.read_csv('/content/drive/MyDrive/lit_met.txt', sep='\t')
df_met = pd.read_csv('/content/drive/MyDrive/df_met_final.txt', sep='\t')

df_literal = df_literal.rename(columns={'Sentence': 'Input'})
df_met = df_met.rename(columns={'Sentence': 'Input'})

# clean output
df_literal['Output'] = df_literal['Output'].apply(clean_output_special)
df_met['Output'] = df_met['Output'].apply(clean_output_special)

df_literal['Input'] = df_literal['Input'].str.strip()
df_literal['Output'] = df_literal['Output'].str.strip()
df_literal = df_literal.dropna(subset=['Input', 'Output'])

df_met['Input'] = df_met['Input'].str.strip()
df_met['Output'] = df_met['Output'].str.strip()
df_met = df_met.dropna(subset=['Input', 'Output'])

fr_df_all = pd.concat([df_literal, df_met], ignore_index=True)
fr_df_all = fr_df_all.dropna()

# Put a placeholder for all the instances where there is no metaphor, as NAN cannot be procesed later
fr_df_all['Output'] = fr_df_all['Output'].replace("", " ")
# pad the ouptuts and ensure there is always a triple
def ensure_triple(data):
    result = []
    for item in data:
        #item = item.replace(",", "|")
        item_list = [x.strip() for x in item.split(",")]
        # If the item is a tuple or listconvert it to a list and check its len
        if len(item_list) < 3:
            item_list.append('')
            # If it has less than 3 elements, add 'nothing' to fill the missing slots
            while len(item_list) < 3:
                item_list.append("")

        item =" ".join(item_list)
        result.append(item)
    return result

fr_df_all['Output'] = ensure_triple(fr_df_all['Output'])

# Preprocessing english data

def clean_output(text):
    # Remove commas, parentheses, and square brackets
    return re.sub(r'[\(\)\[\]\']', '', text)

def clean_output_special(text):
    if pd.isna(text):
        return text
    return re.sub(r'[\[\]\']' , '', text)
df_literal = pd.read_csv('/content/drive/MyDrive/eng_lit_mihan.csv', delimiter=';', encoding='ISO-8859-1')


df_literal['Output'] = df_literal['Output'].apply(clean_output_special)
df_literal['Input'] = df_literal['Input'].str.strip()
df_literal['Output'] = df_literal['Output'].str.strip()
df_literal = df_literal.dropna(subset=['Input', 'Output'])
# size 350
# for index, row in df_cleaned.iterrows():
#     print(index)
#     print(row['Input'])
#     print(row['Output'])


# Output (Subject, Verb, Object For Active voices) and (Object, Verb, Subjet For Passive Voices)
df_met = pd.read_csv('/content/drive/MyDrive/eng_met_mihan.csv', delimiter=';', encoding='UTF-8')

df_met = df_met.rename(columns={'Output (Subject, Verb, Object For Active voices) and (Object, Verb, Subjet For Passive Voices)': 'Output'})
df_met['Output'] = df_met['Output'].apply(clean_output)
df_met['Input'] = df_met['Input'].str.strip()
df_met['Output'] = df_met['Output'].str.strip()
df_met = df_met.drop(columns=['Verb'])
df_met = df_met.dropna(subset=['Input', 'Output'])

# size 350
# Now the other one
eng_all_viet = pd.read_csv('/content/drive/MyDrive/eng_all_viet.csv',delimiter=';', encoding='UTF-8' )
eng_all_viet = eng_all_viet.rename(columns={'Tuple SVO': 'Output'})
eng_all_viet = eng_all_viet.rename(columns={'Sentence': 'Input'})

# swap the columns to match the other dataset
cols = list(eng_all_viet.columns)

# Swap the first two elements
#cols[0], cols[1] = cols[1], cols[0]

# Reorder the DataFrame using the new column order
#eng_all_viet = eng_all_viet[cols]

eng_all_viet['Output'] = eng_all_viet['Output'].apply(clean_output)
empty_count = 0
met_count = 0

for index, row in eng_all_viet.iterrows():
    if row['Output'] == '':
        empty_count += 1
    else:
        met_count += 1


eng_df_all = pd.concat([eng_all_viet, df_met, df_literal], ignore_index=True)


eng_df_all = eng_df_all.dropna()

# Put a placeholder for all the instances where there is no metaphor, as NAN cannot be procesed later
eng_df_all['Output'] = eng_df_all['Output'].replace("", "")
# pad the ouptuts and ensure there is always a triple
def ensure_triple(data):
    result = []
    for item in data:
        #item = item.replace(",", "|")
        item_list = [x.strip() for x in item.split(",")]
        # If the item is a tuple or listconvert it to a list and check its len
        if len(item_list) < 3:
            item_list.append('')
            # If it has less than 3 elements, add 'nothing' to fill the missing slots
            while len(item_list) < 3:
                item_list.append("")

        item =" ".join(item_list)
        result.append(item)
    return result

eng_df_all['Output'] = ensure_triple(eng_df_all['Output'])

#eng_df_all.iloc[707]['Input']  # 415, 3, 700

# Preprocessing german data

def clean_output(text):
    # Remove commas, parentheses, and square brackets
    return re.sub(r'[\(\)\[\]\']', '', text)

def clean_output_special(text):
    if pd.isna(text):
        return text
    return re.sub(r'[\[\]\']' , '', text)

df_literal = pd.read_csv('/content/drive/MyDrive/annotation_nonmet.csv', delimiter=';', header=None)
df_met = pd.read_csv('/content/drive/MyDrive/annotation_met_new_version.csv', delimiter=';', header=None)

df_literal.columns = ['Input', 'Output']
df_met.columns = ['Input', 'Output']

# clean output

df_literal['Input'] = df_literal['Input'].str.strip()
df_literal['Output'] = df_literal['Output'].str.strip()
df_literal = df_literal.dropna(subset=['Input', 'Output'])

df_met['Input'] = df_met['Input'].str.strip()
df_met['Output'] = df_met['Output'].str.strip()
df_met = df_met.dropna(subset=['Input', 'Output'])


df_literal['Output'] = df_literal['Output'].apply(clean_output)
df_met['Output'] = df_met['Output'].apply(clean_output)
#reset indexes

df_met = df_met.reset_index(drop=True)
df_literal = df_literal.reset_index(drop=True)

ger_df_all = pd.concat([df_literal, df_met], ignore_index=True)
ger_df_all = ger_df_all.dropna()

# Put a placeholder for all the instances where there is no metaphor, as NAN cannot be procesed later

ger_df_all['Output'] = ger_df_all['Output'].replace("", " ")
# pad the ouptuts and ensure there is always a triple
def ensure_triple(data):
    result = []
    for item in data:
        #item = item.replace(",", "|")
        item_list = [x.strip() for x in item.split(",")]
        # If the item is a tuple or listconvert it to a list and check its len
        if len(item_list) < 3:
            item_list.append('')
            # If it has less than 3 elements, add 'nothing' to fill the missing slots
            while len(item_list) < 3:
                item_list.append("")

        item =" ".join(item_list)
        result.append(item)
    return result

ger_df_all['Output'] = ensure_triple(ger_df_all['Output'])

# Optimizersss
#1. Optuna

def optuna_hp_space(trial):

    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64, 128]),
    }

#2. SigOpt

def sigopt_hp_space(trial):

    return [
        {"bounds": {"min": 1e-6, "max": 1e-4}, "name": "learning_rate", "type": "double"},
        {
            "categorical_values": ["16", "32", "64", "128"],
            "name": "per_device_train_batch_size",
            "type": "categorical",
        },
    ]

#3. raytune

def ray_hp_space(trial):
    return {
        "learning_rate": tune.loguniform(1e-6, 1e-4),
        "per_device_train_batch_size": tune.choice([16, 32, 64, 128]),
    }

#4.Wandb
def wandb_hp_space(trial):
    return {
        "method": "random",
        "metric": {"name": "objective", "goal": "minimize"},
        "parameters": {
            "learning_rate": {"distribution": "uniform", "min": 1e-6, "max": 1e-4},
            "per_device_train_batch_size": {"values": [16, 32, 64, 128]},
        },
    }

# # include new metrics in class TRAINER, trying lev and cos seperate our normal loss and a combination

# class CustomTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         # Get model predictions
#         outputs = model(**inputs)
#         logits = outputs.get('logits')
#         labels = inputs.get('labels')
#         predictions = torch.argmax(logits, dim=-1)
#         loss_fct = torch.nn.CrossEntropyLoss()
#         loss = loss_fct(logits.view(-1, self.model.config.vocab_size), labels.view(-1))

#         # Convert predictions and labels to text
#         pred_texts = [tokenizer.decode(pred, skip_special_tokens=True).strip() for pred in predictions]
#         label_texts = [tokenizer.decode(label, skip_special_tokens=True).strip() for label in labels]

#         # cos metric
#         cos_loss = 0.0
#         for pred_text, label_text in zip(pred_texts, label_texts):
#             if pred_text and label_text: # check that pred and label are non-empty
#               pred_vec = tokenizer.encode(pred_text, return_tensors='pt')
#               label_vec = tokenizer.encode(label_text, return_tensors='pt')
#               if pred_vec.shape[1] > 0 and label_vec.shape[1] > 0: # check that arrays are non-empty
#                 cos_sim = cosine_similarity(pred_vec.detach().numpy(), label_vec.detach().numpy())
#            # Convert similarity to distances
#                 cos_loss += 1 - cos_sim.mean()
#               else:
#                 # Handle cases where vectors are empty, possibly by assigning a default loss
#                 cos_loss += 1
#             else:
#               cos_loss += 1

#         # Normalize the cos loss
#         cos_loss /= len(pred_texts)

#         #Lev Distance Metric
#         lev_loss = 0.0
#         for pred_text, label_text in zip(pred_texts, label_texts):
#             lev_distance = Levenshtein.distance(pred_text, label_text)
#             max_len = max(len(pred_text), len(label_text))
#             lev_loss += lev_distance / max_len  # Normalize Levenshtein distance by the max length


#         lev_loss /= len(pred_texts)

#         # Combine the original cross-entropy loss with custom metrics
#         combined_loss = loss + cos_loss + lev_loss
#         #combined_loss = cos_loss + loss
#         #combined_loss = lev_loss + loss
#         return (combined_loss, outputs) if return_outputs else combined_loss

# import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def levenshtein_metric(predictions, references):
    distances = []
    for pred, ref in zip(predictions, references):
        pred_text = tokenizer.decode(pred, skip_special_tokens=True)
        ref_text = tokenizer.decode(ref, skip_special_tokens=True)
        # Compute the normalized Lev
        distance = lev.distance(pred_text, ref_text) / max(len(pred_text), len(ref_text))
        distances.append(1 - distance)
    return np.mean(distances)

def token_level_metrics(predictions, references):
    precision_list, recall_list, f1_list = [], [], []
    for pred, ref in zip(predictions, references):
        pred_ids = pred.flatten()
        ref_ids = ref.flatten()

        # Ignore padding
        mask = ref_ids != tokenizer.pad_token_id

        # Precision, recall, f1
        precision = precision_score(ref_ids[mask], pred_ids[mask], average='micro')
        recall = recall_score(ref_ids[mask], pred_ids[mask], average='micro')
        f1 = f1_score(ref_ids[mask], pred_ids[mask], average='micro')

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return np.mean(precision_list), np.mean(recall_list), np.mean(f1_list)

nltk.download('punkt')

# def bleu_score_metric(predictions, references):
#     bleu_scores = []
#     for pred, ref in zip(predictions, references):
#         pred_tokens = tokenizer.decode(pred, skip_special_tokens=True).split()
#         ref_tokens = tokenizer.decode(ref, skip_special_tokens=True).split()

#         # Calculate the BLEU score for each pair of sentences
#         bleu_scores.append(sentence_bleu([ref_tokens], pred_tokens))

#     return np.mean(bleu_scores)
def bleu_score_metric(predictions, references):
    bleu_scores = []
    smoothing_function = SmoothingFunction().method4
    for pred, ref in zip(predictions, references):
        pred_tokens = tokenizer.decode(pred, skip_special_tokens=True).split()
        ref_tokens = tokenizer.decode(ref, skip_special_tokens=True).split()

        # Calculate BLEU for each pair of sentences
        bleu_scores.append(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing_function))

    return np.mean(bleu_scores)

def calculate_rouge(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    levenshtein_sim = levenshtein_metric(predictions, labels)
    precision, recall, f1 = token_level_metrics(predictions, labels)
    return {
        'levenshtein_similarity': levenshtein_sim,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# # Hypertune with IA3: only run Once it takes forever
# class IA3Layer(nn.Module):
#     def __init__(self, hidden_dim):
#         super(IA3Layer, self).__init__()
#         # Learnable scaling vectors for keys, queries, and values
#         self.ia3_key = nn.Parameter(torch.ones(hidden_dim))
#         self.ia3_query = nn.Parameter(torch.ones(hidden_dim))
#         self.ia3_value = nn.Parameter(torch.ones(hidden_dim))

#     def forward(self, query, key, value):
#         print(f"Query shape before IA3: {query.shape}", flush=True)
#         print(f"Key shape before IA3: {key.shape}",  flush=True)
#         print(f"Value shape before IA3: {value.shape}",  flush=True)

#         query = query * self.ia3_query
#         key = key * self.ia3_key
#         value = value * self.ia3_value

#         print(f"Query shape after IA3: {query.shape}",  flush=True)
#         print(f"Key shape after IA3: {key.shape}",  flush=True)
#         print(f"Value shape after IA3: {value.shape}",  flush=True)
#         return query, key, value

# class GPT2AttentionWithIA3(GPT2Attention):
#     def __init__(self, config):
#         super().__init__(config)
#         # Initialize IA3 scaling layers
#         self.ia3 = IA3Layer(config.hidden_size)

#     def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False):
#         # Perform the standard GPT-2 attention mechanism
#         print("1",  flush=True)
#         query, key, value = self._attn(hidden_states, attention_mask, head_mask, use_cache, output_attentions)
#         print("2",  flush=True)
#         query, key, value = self.ia3(query, key, value)
#         print("3",  flush=True)
#         # Rest of attention
#         return super().forward(hidden_states, layer_past, attention_mask, head_mask, use_cache, output_attentions)

# class GPT2WithIA3(GPT2LMHeadModel):
#     def __init__(self, config):
#         super().__init__(config)
#         # Replace the attention layers with IA3-enhanced layers
#         for layer in self.transformer.h:
#             layer.attn = GPT2AttentionWithIA3(config)

# # Load the pre-trained GPT-2 model and replace the attention layers
# config = GPT2Config.from_pretrained("gpt2")
# model = GPT2WithIA3(config)

model_name="antoiloui/belgpt2"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
#config = GPT2Config.from_pretrained("gpt2")
#model = GPT2WithIA3(config)

# Add the padding token to GPT-2's tokenizer
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
def get_layer_weights_hook(module, input, output):
    # Save the output
    module.layer_outputs = output.detach().cpu().numpy()

# Attach the hook to a specific layer
model.lm_head.register_forward_hook(get_layer_weights_hook)
class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=32, max_length_out = 32):
        self.inputs = df['Input'].tolist()
        self.outputs = df['Output'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_length_out= max_length_out

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        output_text = self.outputs[idx]

        tokenized_output = self.tokenizer(output_text, truncation=True, return_tensors='pt', padding='max_length', max_length=self.max_length_out)
        tokenized_input = self.tokenizer(input_text, truncation=True, return_tensors='pt', padding='max_length', max_length=self.max_length)
        #print(tokenized_input['input_ids'].shape, tokenized_output['input_ids'].shape)

        empty_token_id = self.tokenizer.convert_tokens_to_ids('<EMPTY>')
        output_mask = (output_ids != empty_token_id).long()
        model.resize_token_embeddings(len(tokenizer))
        # Return input_ids and attention_mask for training, no labels

        # Input ids and attention mask
        input_ids = tokenized_input['input_ids'].squeeze()  # shape: (max_length)
        attention_mask = tokenized_input['attention_mask'].squeeze()  # shape: (max_length)
        output_ids = tokenized_output['input_ids'].squeeze()  # shape: (max_length)

        # Return input_ids and attention_mask for training, no labels
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': output_ids,
            'output_mask': output_mask
        }

class MaskedTextDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=32, max_length_out = 32):
        self.inputs = df['Input'].tolist()
        self.outputs = df['Output'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_length_out= max_length_out
        self.stored_input_ids = []
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        output_text = self.outputs[idx]

        tokenized_output = self.tokenizer(output_text, truncation=True, return_tensors='pt', padding='max_length', max_length=self.max_length)
        tokenized_input = self.tokenizer(input_text, truncation=True, return_tensors='pt', padding='max_length', max_length=self.max_length)
        #print(tokenized_input['input_ids'].shape, tokenized_output['input_ids'].shape)
        # Input ids and attention mask
        input_ids = tokenized_input['input_ids'].squeeze()  # shape: (max_length)
        attention_mask = tokenized_input['attention_mask'].squeeze()  # shape: (max_length)
        output_ids = tokenized_output['input_ids'].squeeze()  # shape: (max_length)

        # create Mask for empty and delimiter
        #hash_token_id = self.tokenizer.convert_tokens_to_ids('#')
        #delimiter_token_id = self.tokenizer.convert_tokens_to_ids('|')

        #output_mask = (output_ids != hash_token_id).long() & (output_ids != delimiter_token_id).long()
        self.stored_input_ids.append(input_ids.tolist())  # Store the input IDs
        # Return input_ids and attention_mask for training, no labels
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': output_ids
        }

class TextDatasetWithPrompts(Dataset):
    def __init__(self, df, tokenizer, max_length=64, max_length_out=64):
        self.inputs = df['Input'].tolist()
        self.outputs = df['Output'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_length_out = max_length_out

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = f"Input: {self.inputs[idx]}\nOutput:"
        output_text = self.outputs[idx]

        tokenized_input = self.tokenizer(
            input_text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt'
        )
        tokenized_output = self.tokenizer(
            output_text, truncation=True, padding='max_length', max_length=self.max_length_out, return_tensors='pt'
        )

        input_ids = tokenized_input['input_ids'].squeeze()
        attention_mask = tokenized_input['attention_mask'].squeeze()
        output_ids = tokenized_output['input_ids'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': output_ids
        }

# Load data into the custom dataset
My_dataset = MaskedTextDataset(df=fr_df_all, tokenizer=tokenizer)
# Split into training and test sets
split = 0.8
train_eval_size = int(split * len(My_dataset))
test_size = len(My_dataset) - train_eval_size
train_eval_data, test_data = random_split(My_dataset, [train_eval_size, test_size])

train_size = int(split * len(train_eval_data))
eval_size = len(train_eval_data) - train_size
training_data, eval_data = random_split(train_eval_data, [train_size, eval_size])

# Define DataLoaders
batch_size = 64
epochs = 50
# dataloader_train = DataLoader(training_data, batch_size=batch_size, shuffle=True)
# dataloader_eval = DataLoader(eval_data, batch_size=batch_size, shuffle=True)
# dataloader_test = DataLoader(test_data, batch_size=batch_size, shuffle=True)
######
# masking outside
#special_tokens_dict = {'additional_special_tokens': ['#']}
#tokenizer.add_special_tokens(special_tokens_dict)

class LogInputIDsCallback(TrainerCallback):
    def __init__(self, experiment, tokenizer, My_dataset):
        self.experiment = experiment
        self.tokenizer = tokenizer
        self.dataset = My_dataset
    def on_log(self, args, state, control, **kwargs):
        step = state.global_step
        print(step)
        # Get the corresponding input_ids from the dataset
        input_ids = self.dataset.stored_input_ids[-step:]
        print(input_ids)
        if input_ids is not None:
              for ids in input_ids:
                  tokens = self.tokenizer.convert_ids_to_tokens(ids)
                  tokens = self.tokenizer.decode(tokens)
                  token_str = " ".join(tokens)
                  print(f"Logging tokens for step {step}: {token_str}")

                  # Log to Comet
                  self.experiment.log_text(f"Tokens: {token_str}", metadata={"step": step})
######
# Define training args
training_args = TrainingArguments(
    logging_steps=50,
    output_dir='./results-small',
    learning_rate = 5e-4,
    weight_decay = 0.01,
    gradient_accumulation_steps=3,
    overwrite_output_dir=True,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    eval_strategy="epoch",
    report_to="comet_ml"  # This enables reporting to Comet
)

# Model Init for hyperparam optimization

# model_args = {
#     "model": model_name,
#     "from_tf": False,
#     "config": AutoConfig.from_pretrained(model_name),
#     "cache_dir": None, # You can specify a cache directory if needed
#     "revision": None, # You can specify a model revision if needed
#     "token": True # Set to True if using an authentication token
# }

# def model_init(trial):
#     model = GPT2LMHeadModel.from_pretrained(
#         model_name,
#         ignore_mismatched_sizes=True,
#         from_tf=bool(".ckpt" in model_name),
#         config= AutoConfig.from_pretrained(model_name),
#         cache_dir=None,
#         revision=None,
#         token=True,
#     )
#     model.config.pad_token_id = tokenizer.eos_token_id
#     return model

#IA3 init config
peft_config = IA3Config(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["attn.c_attn","attn.c_proj", "mlp.c_proj","mlp.c_fc"],
    feedforward_modules=["mlp.c_fc"],
    modules_to_save=["lm_head"],  # List of trainable modules to save
)
model.resize_token_embeddings(len(tokenizer))
print(model)
model = get_peft_model(model, peft_config)
# print paraneters
model.print_trainable_parameters()
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=training_data,
    eval_dataset=eval_data,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# #Initialize the Trainer with wandb stuff
# trainer = Trainer(
#     model_init=model_init, # This is for the model init for the hyperparm optim
#     model=model,
#     args=training_args,
#     train_dataset=training_data,
#     eval_dataset=eval_data
# )
# trainer = CustomTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=training_data,
#     eval_dataset=eval_data,
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics  # Optionally include custom metrics
# )

#
# ## Get best hyperparams
# best_trials = trainer.hyperparameter_search(
#     direction=["minimize", "maximize"],
#     backend="optuna",
#     hp_space=optuna_hp_space,
#     n_trials=10
# )

# Start finetuning
start_time = time.time()
print('Training starts')
trainer.train()
model.save_pretrained("/content/drive/MyDrive/gpt-small-frfr-finetuned")
#model.save_weights("/content/drive/MyDrive/weights.h5")
print('Training done')
end_time = time.time()
experiment.end()

fine_tuning_time = end_time - start_time
print(f"Training completed in {fine_tuning_time:.2f} seconds.")
trainer.evaluate()

predictions, labels, metrics = trainer.predict(test_dataset=test_data)

predicted_token_ids = np.argmax(predictions, axis=-1)

# Step 2: Decode the predicted token IDs to text
decoded_predictions = [tokenizer.decode(pred_seq, skip_special_tokens=True) for pred_seq in predicted_token_ids]
decoded_labels = [tokenizer.decode(label_seq, skip_special_tokens=True) for label_seq in labels]

# human comparison:
for label,prediction in zip(decoded_labels,decoded_predictions):
  print(f'Label:{label}, Prediction:{prediction}')



# Transfer learning script

# Load data into the custom dataset
My_dataset_eng = MaskedTextDataset(df=eng_df_all, tokenizer=tokenizer)
# Split
split = 0.8
train_eval_size_eng = int(split * len(My_dataset_eng))
test_size_eng = len(My_dataset_eng) - train_eval_size_eng
train_eval_data_eng, test_data_eng = random_split(My_dataset_eng, [train_eval_size_eng, test_size_eng])

My_dataset_ger = MaskedTextDataset(df=ger_df_all, tokenizer=tokenizer)
train_eval_size_ger = int(split * len(My_dataset_ger))
test_size_ger = len(My_dataset_ger) - train_eval_size_ger
train_eval_data_ger, test_data_ger = random_split(My_dataset_ger, [train_eval_size_ger, test_size_ger])

predictions, labels, metrics = trainer.predict(test_dataset=test_data_eng)

predicted_token_ids = np.argmax(predictions, axis=-1)

# Step 2: Decode the predicted token ids
decoded_predictions = [tokenizer.decode(pred_seq, skip_special_tokens=True) for pred_seq in predicted_token_ids]
decoded_labels = [tokenizer.decode(label_seq, skip_special_tokens=True) for label_seq in labels]
# human comparison is here:
for label,prediction in zip(decoded_labels,decoded_predictions):
  print(f'Label:{label}, Prediction:{prediction}')

print(metrics)

predictions, labels, metrics = trainer.predict(test_dataset=test_data_ger)

predicted_token_ids = np.argmax(predictions, axis=-1)

# Step 2: Decode the predicted token IDs to text
decoded_predictions = [tokenizer.decode(pred_seq, skip_special_tokens=True) for pred_seq in predicted_token_ids]
decoded_labels = [tokenizer.decode(label_seq, skip_special_tokens=True) for label_seq in labels]
# human comparison:
for label,prediction in zip(decoded_labels,decoded_predictions):
  print(f'Label:{label}, Prediction:{prediction}')

print(metrics)

