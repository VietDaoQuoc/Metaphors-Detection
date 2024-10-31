# -*- coding: utf-8 -*-
"""german_gpt_training.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1OTXJhn6p-Z4_DsmD9NJRFTB0xm2sElEA
"""

# prep for mount
from google.colab import drive
drive.mount('/content/drive')

!python -m pip install wandb -Uq
!python -m pip install ray[tune]
!python -m pip install sigopt
!python -m pip install optuna

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoConfig, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader, random_split
#import Levenshtein as lev
import time
import logging
import numpy as np
import wandb
import random
import math

!nvidia-smi command

wandb.login() #enter this token: a5d8e3b6d2ef7d55f930ab72670aaa64e1a4198d

!ls '/content/drive/MyDrive'

import pandas as pd
import re
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
df_literal

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
ger_df_all['Output'] = ger_df_all['Output'].replace("", "#,#,#")
# pad the ouptuts and ensure there is always a triple

def ensure_triple(data):
    result = []
    for item in data:
        item = item.replace(",", "|")
        item_list = [x.strip() for x in item.split("|")]
        # If the item is a tuple or list, convert it to a list and check its length
        if len(item_list) < 3:
            item_list.append('#')
            # If it has less than 3 elements, add 'nothing' to fill the missing slots
            while len(item_list) < 3:
                item_list.append("#")
        item ="|".join(item_list)
        result.append(item)
    return result

ger_df_all['Output'] = ensure_triple(ger_df_all['Output'])
ger_df_all

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

model_name='stefan-it/german-gpt2-larger'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Add the padding token to GPT-2's tokenizer (optional, but useful)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
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
        # Input IDs and attention mask
        input_ids = tokenized_input['input_ids'].squeeze()  # shape: (max_length)
        attention_mask = tokenized_input['attention_mask'].squeeze()  # shape: (max_length)
        output_ids = tokenized_output['input_ids'].squeeze()  # shape: (max_length)

        # Return input_ids and attention_mask for training, no labels
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': output_ids
        }

class MaskedTextDataset(Dataset):
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

        tokenized_output = self.tokenizer(output_text, truncation=True, return_tensors='pt', padding='max_length', max_length=self.max_length)
        tokenized_input = self.tokenizer(input_text, truncation=True, return_tensors='pt', padding='max_length', max_length=self.max_length)
        #print(tokenized_input['input_ids'].shape, tokenized_output['input_ids'].shape)
        # Input IDs and attention mask
        input_ids = tokenized_input['input_ids'].squeeze()  # shape: (max_length)
        attention_mask = tokenized_input['attention_mask'].squeeze()  # shape: (max_length)
        output_ids = tokenized_output['input_ids'].squeeze()  # shape: (max_length)

        # create Mask for empty and delimiter
        hash_token_id = self.tokenizer.convert_tokens_to_ids('#')
        delimiter_token_id = self.tokenizer.convert_tokens_to_ids('|')
        output_mask = (output_ids != hash_token_id).long() & (output_ids != delimiter_token_id).long()
        model.resize_token_embeddings(len(tokenizer))
        # Return input_ids and attention_mask for training, no labels
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': output_ids,
            'output_mask': output_mask  # Add the output mask
        }

# Load data into the custom dataset
My_dataset = MaskedTextDataset(df=ger_df_all, tokenizer=tokenizer)
# Split into training and test sets
split = 0.8
train_eval_size = int(split * len(My_dataset))
test_size = len(My_dataset) - train_eval_size
train_eval_data, test_data = random_split(My_dataset, [train_eval_size, test_size])

train_size = int(split * len(train_eval_data))
eval_size = len(train_eval_data) - train_size
training_data, eval_data = random_split(train_eval_data, [train_size, eval_size])

# Define DataLoaders
batch_size = 16
epochs = 500
# dataloader_train = DataLoader(training_data, batch_size=batch_size, shuffle=True)
# dataloader_eval = DataLoader(eval_data, batch_size=batch_size, shuffle=True)
# dataloader_test = DataLoader(test_data, batch_size=batch_size, shuffle=True)
######
# masking outside
#special_tokens_dict = {'additional_special_tokens': ['#']}
#tokenizer.add_special_tokens(special_tokens_dict)
#model.resize_token_embeddings(len(tokenizer))

######
# Define training arguments
training_args = TrainingArguments(
    logging_steps=50,
    output_dir='./results',
    learning_rate = 5e-5,
    weight_decay = 0.05,
    gradient_accumulation_steps=1,
    overwrite_output_dir=True,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    eval_strategy="epoch",
    report_to="none"
)

# Model Init for hyperparam optimization


model_args = {
    "model": model_name,
    "from_tf": False,
    "config": AutoConfig.from_pretrained(model_name),
    "cache_dir": None, # You can specify a cache directory if needed
    "revision": None, # You can specify a model revision if needed
    "token": True # Set to True if using an authentication token
}

def model_init(trial):
    model = GPT2LMHeadModel.from_pretrained(
        model_name,
        ignore_mismatched_sizes=True,
        from_tf=bool(".ckpt" in model_name),
        config= AutoConfig.from_pretrained(model_name),
        cache_dir=None,
        revision=None,
        token=True,
    )
    model.config.pad_token_id = tokenizer.eos_token_id
    return model
# Initialize the Trainer
trainer = Trainer(
    model_init=model_init, # This is for the model init for the hyperparm optim
    model=model,
    args=training_args,
    train_dataset=training_data,
    eval_dataset=eval_data
)

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

## Get best hyperparams
best_trials = trainer.hyperparameter_search(
    direction=["minimize", "maximize"],
    backend="optuna",
    hp_space=optuna_hp_space,
    n_trials=20
)

# Start finetuning
start_time = time.time()
print('Training starts')
trainer.train()
model.save_pretrained("./gpt-french-finetuned")
print('Training done')
end_time = time.time()

fine_tuning_time = end_time - start_time
print(f"Training completed in {fine_tuning_time:.2f} seconds.")
trainer.evaluate()

predictions, labels, metrics = trainer.predict(test_dataset=test_data)

predicted_token_ids = np.argmax(predictions, axis=-1)

# Step 2: Decode the predicted token IDs to text
decoded_predictions = [tokenizer.decode(pred_seq, skip_special_tokens=True) for pred_seq in predicted_token_ids]
decoded_predictions

decoded_labels = [tokenizer.decode(label_seq, skip_special_tokens=True) for label_seq in labels]
decoded_labels

# human comparison:
for label,prediction in zip(decoded_labels,decoded_predictions):
  print(f'Label:{label}, Prediction:{prediction}')

