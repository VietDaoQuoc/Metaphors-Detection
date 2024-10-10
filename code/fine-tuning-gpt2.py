# code block for GPT-2
# using Prefix-tuning for generative models like GPT-2
# without using FrameBert embedding

from transformers import Trainer, TrainingArguments, GPT2Tokenizer, GPT2LMHeadModel
import pandas as pd
from peft import PrefixTuningConfig, get_peft_model

def get_fine_tuned_gpt2():

    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    dataset = pd.read_csv("........")

    train_dataset = tokenizer.encode(dataset['sentences'])
    test_dataset = 'something'

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
   
    prefix_tuning_config = PrefixTuningConfig(
        task_type="CAUSAL_LM",  # GPT-2 is a causal language model
        num_virtual_tokens=20    # Number of prefix tokens to learn
    )

    model = get_peft_model(model, prefix_tuning_config)

    # copied from online source
    training_args = TrainingArguments(
        output_dir="./gpt2-prefix-tuning",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        fp16=True,  
        report_to="none"  
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

 
    trainer.train()


    model.save_pretrained("./gpt2-prefix-tuned")
    tokenizer.save_pretrained("./gpt2-prefix-tuned")

def gpt2_validate():
    model_name = "gpt2"
    model = GPT2LMHeadModel._load_pretrained_model("./gpt2-prefix-tuned")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    model = ...
    max_length = ...
    tokenizer = ...
    pad_token_id = ...
    

    query_input = "I just had a beef with my homies" 
    query_output = ('I', 'have ', 'beef') 
    
    sample_1 = "The wind whispered secrets through the trees."
    sample_2 = "The message then killed itself."
    sample_3 = "Time flew away from her grasp."

    output_1 = "(The wind, whisper, secrets)"
    output_2 = "(The message, kill, itself)"
    output_3 = "(Time, fly, away)"

    prompt = (
        "Extract the (Subject, Verb, Object) tuple from the following metaphorical sentences:\n\n"
        f"Sentence 1: {sample_1}\nTuple: {output_1}\n\n"
        f"Sentence 2: {sample_2}\nTuple: {output_2}\n\n"
        f"Sentence 3: {sample_3}\nTuple: {output_3}\n\n"
        f"Sentence: {query_input}\nTuple:"
    )

    inputs = tokenizer.encode(prompt, return_tensors="pt")

    outputs = model.generate(
        inputs, 
        max_length=max_length, 
        num_return_sequences=1, 
        pad_token_id=pad_token_id 
    )

    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    generated_text = decoded_output.split("Tuple:")[-1].strip()

    return generated_text