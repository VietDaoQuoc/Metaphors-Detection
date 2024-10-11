import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForQuestionAnswering, Trainer, TrainingArguments
import pandas as pd
import time
from sklearn.model_selection import train_test_split

data = pd.read_csv('....')
#split into 80% train+val and 20% test
train_data, test_df = train_test_split(data, test_size=0.2, random_state=42)

#split 80% into 70% of 80% TRAIN and 30% of 80% VAL
train_data, val_data = train_test_split(train_data, test_size=0.3, random_state=42)

def fine_tune(train_data, val_data):
    start_time = time.time()

    model_name = "xlm-roberta-base"
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
    model = XLMRobertaForQuestionAnswering.from_pretrained(model_name, num_labels=2)

    encoded_train = tokenizer.encode(train_data['Sentence'])
    encoded_val = tokenizer.encode(val_data['Sentence'])

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_train,
        eval_dataset=encoded_val,
    )

    print('Training starts ....')
    trainer.train() 
    model.save_pretrained("./xlmroberta-finetuned")
    print('Training done .....')

    end_time = time.time()

    fine_tuning_time = end_time - start_time
    print(f"Training completed in {fine_tuning_time:.2f} seconds.")

def zero_shot_test():

    model_name="xlm-roberta-base"
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)

    model = XLMRobertaForQuestionAnswering.from_pretrained(model_name)

    # question that already appears in training/val data
    question, target = """What is the metaphorical SVO tuple in this sentence: Mr. Reagan , in his first term , tried to kill the agency""", "(Mr. Reagan, kill, agency)"


    inputs = tokenizer(question, target, return_tensors="pt")

    with torch.no_grad():

        outputs = model(**inputs)

    answer_start_index = outputs.start_logits.argmax()

    answer_end_index = outputs.end_logits.argmax()

    predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]

    tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)


    outputs = model(**inputs)
    print(tokenizer.decode(outputs))

def infer_after_finetune(test_df):
    return