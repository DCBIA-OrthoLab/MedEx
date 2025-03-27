import os
import re
import csv
import fitz
import torch
import argparse
import evaluate
import numpy as np
import pandas as pd
from docx import Document
from datasets import load_dataset
from sklearn.model_selection import train_test_split, KFold
from utilities import initialize_key_value_summary
from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq, EarlyStoppingCallback

from utilities import extract_text_from_pdf, extract_text_from_word, create_chunks_from_paragraphs, clean_text

def load_text(file_path):
    """
    Reads and cleans text from a file (PDF, DOCX, or TXT format).

    Args:
        file_path (str): Path to the input file

    Returns:
        str: The cleaned text content of the file

    Raises:
        ValueError: If the file format is unsupported
    """
    if file_path.endswith(".pdf"):
        return clean_text(extract_text_from_pdf(file_path))
    elif file_path.endswith(".docx"):
        return clean_text(extract_text_from_word(file_path))
    elif file_path.endswith(".txt"):
        with open(file_path, 'r', encoding='utf-8') as file:
            return clean_text(file.read())
    else:
        raise ValueError("Unsupported file format. Supported formats: .txt, .pdf, .docx")

def save_notes_and_summaries_to_csv(notes_folder, summaries_folder, output_csv, max_chunk_size=3500):
    """
    Processes clinical notes and summaries into chunked pairs saved as CSV.

    Args:
        notes_folder (str): Directory containing clinical notes
        summaries_folder (str): Directory containing corresponding summaries
        output_csv (str): Path to save the output CSV file
        max_chunk_size (int, optional): Maximum size for text chunks. Defaults to 3500.
    """
    texts, summaries = [], []  # Corrected initialization
    
    for note_filename in os.listdir(notes_folder):
        if note_filename.endswith((".txt", ".pdf", ".docx")):
            patient_id = note_filename.split("_")[0]
            summary_filename = f"{patient_id}_summary.txt"
            note_path, summary_path = os.path.join(notes_folder, note_filename), os.path.join(summaries_folder, summary_filename)
            
            if os.path.exists(note_path) and os.path.exists(summary_path):
                dict = initialize_key_value_summary()
                keys = list(dict.keys())
                
                text = load_text(note_path)
                summary = load_text(summary_path)
                
                chunks = create_chunks_from_paragraphs(text, max_chunk_size=max_chunk_size)
                summary_chunks = split_summary_text(summary)
                
                if len(chunks) != len(summary_chunks):
                    print(f"Warning: Mismatch between text chunks and summary chunks for {patient_id}")
                    print(f"Text chunks: {len(chunks)}, Summary chunks: {len(summary_chunks)}")
                
                for chunk, summary_chunk in zip(chunks, summary_chunks):
                    texts.append(chunk)
                    summaries.append(summary_chunk)

    data = pd.DataFrame({"text": texts, "summary": summaries})
    data.to_csv(output_csv, index=False, quoting=csv.QUOTE_ALL)
    print(f"Data saved to {output_csv}")

def split_summary_text(summary_text):
    """
    Splits a summary into chunks based on a long delimiter pattern.

    Args:
        summary_text (str): The full summary text to split

    Returns:
        list: List of individual summary chunks
    """
    return [chunk.strip() for chunk in summary_text.split("-"*100) if chunk.strip()]

def assign_patient_ids(data):
    """
    Propagates patient IDs through a DataFrame based on extracted identifiers.

    Args:
        data (pd.DataFrame): DataFrame containing clinical text data

    Returns:
        pd.DataFrame: DataFrame with added patient_id column
    """
    current_patient_id = None
    patient_ids = []
    
    for index, row in data.iterrows():
        extracted_id = pd.Series(row['summary']).str.extract(r'patient_id: (\w+)')[0].values[0]
        
        if pd.notna(extracted_id):
            current_patient_id = extracted_id
        
        patient_ids.append(current_patient_id)
    
    data["patient_id"] = patient_ids
    return data

def prepare_folds(input_csv, output_dir, n_splits=5):
    """
    Creates k-fold train/validation/test splits while preserving patient groupings.

    Args:
        input_csv (str): Path to source CSV data
        output_dir (str): Directory to save fold datasets
        n_splits (int, optional): Number of folds to create. Defaults to 5.
    """
    
    data = pd.read_csv(input_csv)
    data = assign_patient_ids(data)  # Ensure patient IDs are assigned correctly
    data = data.dropna(subset=["patient_id"])
    
    unique_patients = np.array(data["patient_id"].unique())

    for fold in range(n_splits):
        print(f"Preparing fold {fold + 1}/{n_splits}")

        # Step 1: Shuffle the unique patient IDs
        np.random.seed(fold)  # Ensures different but reproducible splits per fold
        np.random.shuffle(unique_patients)

        # Step 2: Split patients into 80%-10%-10%
        train_patients, temp_patients = train_test_split(unique_patients, train_size=0.80, random_state=fold)
        val_patients, test_patients = train_test_split(temp_patients, train_size=0.50, random_state=fold)

        # Step 3: Extract corresponding data
        train_data = data[data["patient_id"].isin(train_patients)]
        val_data = data[data["patient_id"].isin(val_patients)]
        test_data = data[data["patient_id"].isin(test_patients)]

        # Remove patient ID before saving
        train_data = train_data.drop(columns=["patient_id"])
        val_data = val_data.drop(columns=["patient_id"])
        test_data = test_data.drop(columns=["patient_id"])

        # Create fold directory
        fold_dir = os.path.join(output_dir, f"fold_{fold + 1}")
        os.makedirs(fold_dir, exist_ok=True)

        # Save train, validation, and test CSVs for this fold
        train_data.to_csv(os.path.join(fold_dir, "train.csv"), index=False, quoting=csv.QUOTE_ALL)
        val_data.to_csv(os.path.join(fold_dir, "validation.csv"), index=False, quoting=csv.QUOTE_ALL)
        test_data.to_csv(os.path.join(fold_dir, "test.csv"), index=False, quoting=csv.QUOTE_ALL)

        print(f"Fold {fold + 1} created: Train={len(train_data)}, Validation={len(val_data)}, Test={len(test_data)}")

    print(f"Cross-validation folds prepared and saved in {output_dir}")

def fine_tune(training_path, validation_path, output_dir):
    """
    Fine-tunes a BART model for summarization on provided datasets.

    Args:
        training_path (str): Path to training CSV
        validation_path (str): Path to validation CSV
        output_dir (str): Directory to save trained model
    """
    dataset = load_dataset("csv", data_files={"train": training_path, "validation": validation_path})

    model_name = "facebook/bart-large-cnn"
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)
    
    keys = list(initialize_key_value_summary().keys())

    def preprocess_function(examples):
        """
        Preprocesses text-summary pairs into tokenized inputs for the BART model.
    
        Args:
            examples (dict): A batch containing:
                - "text" (list[str]): List of input texts
                - "summary" (list[str]): List of target summaries
    
        Returns:
            dict: Processed batch containing:
                - "input_ids": Tokenized input texts
                - "attention_mask": Attention masks for inputs
                - "labels": Tokenized target summaries
        """
        input_texts = [f'Using this list: {keys}, summarize this note: {text}' for text in examples["text"]]
        model_inputs = tokenizer(examples["text"], max_length=1024, truncation=True, padding="max_length")
        labels = tokenizer(examples["summary"], max_length=150, truncation=True, padding="max_length").input_ids
        model_inputs["labels"] = labels
        return model_inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=20,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_rouge1",
        greater_is_better=True,
    )

    def compute_metrics(eval_preds):
        """
        Computes ROUGE scores between generated and reference summaries.
    
        Args:
            eval_preds (tuple): Contains:
                - logits (np.ndarray): Model prediction logits
                - labels (np.ndarray): Ground truth label IDs
    
        Returns:
            dict: ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum) scaled to 0-100
        """
        metric = evaluate.load("rouge")
        logits, labels = eval_preds
        
        if isinstance(logits, tuple):
            logits = logits[0]
            
        predictions = np.argmax(logits, axis=-1)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {key: value * 100 for key, value in result.items()}
        return result

    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=20)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback],
    )

    trainer.train()
    results = trainer.evaluate()
    print(results)

    model.config.forced_bos_token_id = 0 
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

def cross_validate(csv_folder, output_dir, n_splits=5):
    """
    Performs k-fold cross-validation for model evaluation.

    Args:
        csv_folder (str): Directory containing pre-split fold data
        output_dir (str): Directory to save model outputs
        n_splits (int, optional): Number of folds. Defaults to 5.

    Returns:
        dict: Aggregated evaluation metrics across all folds
    """
    fold_results = []
    
    for fold in range(n_splits):
        print(f"Processing fold {fold + 1}/{n_splits}")
        
        # Define paths for training and validation CSV files for this fold
        train_csv = os.path.join(csv_folder, f"fold_{fold + 1}", "train.csv")
        val_csv = os.path.join(csv_folder, f"fold_{fold + 1}", "validation.csv")
        
        # Fine-tune the model on this fold
        model_dir = os.path.join(output_dir, f"fold_{fold + 1}", "model")
        os.makedirs(model_dir, exist_ok=True)
        
        fine_tune(
            training_path=train_csv,
            validation_path=val_csv,
            output_dir=model_dir
        )
        
        # Evaluate the model on the test set
        test_csv = os.path.join(csv_folder, f"fold_{fold + 1}", "test.csv")
        test_results = evaluate_model(model_dir, test_csv)
        fold_results.append(test_results)
        print(f"Fold {fold + 1} results: {test_results}")
    
    # Aggregate results
    aggregated_results = aggregate_results(fold_results)
    print("Aggregated results:", aggregated_results)

def evaluate_model(model_dir, test_csv):
    """
    Evaluates a trained model on test data.

    Args:
        model_dir (str): Directory containing saved model
        test_csv (str): Path to test CSV data

    Returns:
        dict: Evaluation metrics
    """
    model = BartForConditionalGeneration.from_pretrained(model_dir)
    tokenizer = BartTokenizer.from_pretrained(model_dir)
    
    test_dataset = load_dataset("csv", data_files={"test": test_csv})
    
    def preprocess_function(examples):
        """
        Tokenizes text-summary pairs without additional prompt formatting.
    
        Args:
            examples (dict): A batch containing:
                - "text" (list[str]): Input texts
                - "summary" (list[str]): Target summaries
    
        Returns:
            dict: Tokenized inputs with:
                - "input_ids": Tokenized text
                - "attention_mask": Input masks
                - "labels": Tokenized summaries
        """
        model_inputs = tokenizer(examples["text"], max_length=1024, truncation=True, padding="max_length")
        labels = tokenizer(examples["summary"], max_length=150, truncation=True, padding="max_length").input_ids
        model_inputs["labels"] = labels
        return model_inputs
    
    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)
    
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=model_dir, per_device_eval_batch_size=8),
        eval_dataset=tokenized_test_dataset["test"],
        tokenizer=tokenizer,
    )
    
    results = trainer.evaluate()
    return results

def aggregate_results(fold_results):
    """
    Computes mean metrics across cross-validation folds.

    Args:
        fold_results (list): List of evaluation metrics per fold

    Returns:
        dict: Averaged metrics across all folds
    """
    aggregated_results = {}
    for key in fold_results[0].keys():
        aggregated_results[key] = np.mean([result[key] for result in fold_results])
    return aggregated_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data or fine-tune model for summarization.")
    parser.add_argument("--prepare_data", action="store_true", help="If set, prepares the data by saving clinical notes to CSV.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with clinical notes with prompts in .txt format.")
    parser.add_argument("--summaries_dir", type=str, required=True, help="Directory with summaries in .txt format.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output model or CSV.")
    parser.add_argument("--CSV_folder", type=str, help="Folder containing pre-split CSV files for cross-validation.")
    parser.add_argument("--cross_validate", action="store_true", help="If set, performs 5-fold cross-validation.")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.prepare_data:
        output_csv = os.path.join(args.output_dir, "clinical_data.csv")
        save_notes_and_summaries_to_csv(args.input_dir, args.summaries_dir, output_csv)
        prepare_folds(output_csv, args.output_dir)
    elif args.cross_validate:
        if not args.CSV_folder:
            raise ValueError("Please specify --CSV_folder for cross-validation.")
        cross_validate(args.CSV_folder, args.output_dir)
    else:
        if args.train_csv and args.val_csv:
            fine_tune(args.train_csv, args.val_csv, args.output_dir)
        else:
            print("Error: Provide both --train_csv and --val_csv for fine-tuning.")
