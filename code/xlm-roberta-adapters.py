from transformers import XLMRobertaTokenizer, XLMRobertaAdapterModel, Trainer, TrainingArguments
import pandas as pd
import time
from sklearn.model_selection import train_test_split

model_name = "xlm-roberta-base"
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
model = XLMRobertaAdapterModel.from_pretrained(model_name)

# Add a French language adapter
language_adapter_name = "fr"
model.add_adapter(language_adapter_name)
model.train_adapter(language_adapter_name)  # Freeze the main model, only train the adapter
