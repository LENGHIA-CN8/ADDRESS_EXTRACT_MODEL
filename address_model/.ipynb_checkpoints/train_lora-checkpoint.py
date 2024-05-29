import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import(
    AdamW,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from datasets import Dataset
import argparse
import evaluate
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType

rouge = evaluate.load("rouge")

def get_device_and_set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device

def preprocess_function(examples, tokenizer):
    model_inputs = tokenizer(
        examples["inputs"], max_length=256, truncation=True, padding=True
    )
    
    
    labels = tokenizer(
        examples["labels"], max_length=256, truncation=True, padding=True
    )
    model_inputs['labels'] = labels['input_ids']
    model_inputs['input_ids'] = model_inputs['input_ids']
    return model_inputs
    
SEED = 123
device = get_device_and_set_seed(SEED)
print(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training model")
    
    parser.add_argument("--model_name", type=str, default="default_model", help="Name of the model")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to store model ckpt")

    # Parse command-line arguments
    args = parser.parse_args()
    
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, load_in_8bit=True, device_map="auto")
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast = True)  
    data_df = pd.read_csv(args.dataset_path)
    meta_df = pd.read_csv('./data/address_data_meta_1.csv')
    
    train_df, test_df = train_test_split(data_df, test_size=0.1, random_state=SEED)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=SEED)
    train_df = pd.concat([train_df, meta_df], ignore_index=True)
    val_df = val_df.reset_index(drop=True)
    
    print(f'Total train data {len(train_df)}')
    print(f'Total val data {len(val_df)}')
    print(f'Total test data {len(test_df)}')
    dict_obj = {'inputs': train_df['mistake_address'], 'labels': train_df['filter_address']}
    dataset = Dataset.from_dict(dict_obj)
    # dataset = dataset.train_test_split(test_size=0.1)
    train_data = dataset.map(preprocess_function, batched=True, remove_columns=['inputs'], num_proc=8, fn_kwargs={"tokenizer": tokenizer})

    dict_obj = {'inputs': val_df['mistake_address'], 'labels': val_df['filter_address']}
    dataset = Dataset.from_dict(dict_obj)
    val_data = dataset.map(preprocess_function, batched=True, remove_columns=['inputs'], num_proc=8, fn_kwargs={"tokenizer": tokenizer})
    
    dict_obj = {'inputs': test_df['mistake_address'], 'labels': test_df['filter_address']}
    dataset = Dataset.from_dict(dict_obj)
    test_data = dataset.map(preprocess_function, batched=True, remove_columns=['inputs'], num_proc=8, fn_kwargs={"tokenizer": tokenizer})
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8, return_tensors="pt")
    
    # Define LoRA Config
    lora_config = LoraConfig(
     r=8,
     lora_alpha=16,
     target_modules=["q", "v"],
     lora_dropout=0.05,
     bias="none",
     task_type=TaskType.SEQ_2_SEQ_LM
    )
    # prepare int-8 model for training
    model = prepare_model_for_int8_training(model)

    # add LoRA adaptor
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    training_args = Seq2SeqTrainingArguments(
    output_dir=args.output_dir,
    evaluation_strategy='steps',
    per_device_train_batch_size=args.per_device_train_batch_size,
    learning_rate=1e-5, # higher learning rate
    num_train_epochs=args.epochs,
    logging_dir=f"{args.output_dir}/logs",
    logging_steps = 1000,
    save_steps = 0.05,
    eval_steps = 0.05,
    group_by_length=True,
    save_strategy='steps',
    load_best_model_at_end=True,
    save_total_limit=2,
    )
    
    trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
    )

    trainer.train()

