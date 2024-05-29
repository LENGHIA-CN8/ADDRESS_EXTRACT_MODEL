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

rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

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
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")

    # Parse command-line arguments
    args = parser.parse_args()
    
    model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-base")
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")  
    data_df = pd.read_csv(args.dataset_path)
    
    train_df, test_df = train_test_split(data_df, test_size=0.1, random_state=SEED)
    
    dict_obj = {'inputs': train_df['mistake_address'], 'labels': train_df['filter_address']}
    dataset = Dataset.from_dict(dict_obj)
    dataset = dataset.train_test_split(test_size=0.1)
    train_data = dataset.map(preprocess_function, batched=True, remove_columns=['inputs'], num_proc=8, fn_kwargs={"tokenizer": tokenizer})
    
    dict_obj = {'inputs': test_df['mistake_address'], 'labels': test_df['mistake_address']}
    dataset = Dataset.from_dict(dict_obj)
    test_data = dataset.map(preprocess_function, batched=True, remove_columns=['inputs'], num_proc=8, fn_kwargs={"tokenizer": tokenizer})
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt")
    
    training_args = Seq2SeqTrainingArguments(
    "T5_address_model/",
    do_train=True,
    do_eval=True,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    num_train_epochs=15,
    learning_rate=1e-5,
    warmup_ratio=0.05,
    weight_decay=0.01,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=2,
    logging_dir='./log',
    group_by_length=True,
    load_best_model_at_end=True,
    save_total_limit=1,
    fp16=True,
    )
    
    trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data["train"],
    eval_dataset=train_data["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
    )

    trainer.train()

