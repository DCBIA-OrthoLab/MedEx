import argparse
import os
import re
import json
import torch
import string
from transformers import BartTokenizer, BartForConditionalGeneration, GenerationConfig
from utilities import initialize_key_value_summary, create_chunks_from_paragraphs, extract_text_from_pdf, extract_text_from_word, load_model_and_tokenizer

def generate_combined_summary(model, tokenizer, text, max_chunk_size=3500, model_max_tokens=1024):
    """
    Generates a combined summary by processing text in chunks using a BART model.

    Args:
        model (BartForConditionalGeneration): Pre-trained BART model for summarization
        tokenizer (BartTokenizer): Tokenizer for the BART model
        text (str): Input text to be summarized
        max_chunk_size (int, optional): Maximum character length for each chunk. Defaults to 3500.
        model_max_tokens (int, optional): Maximum token length for model input. Defaults to 1024.

    Returns:
        str: Combined summary of all chunks separated by dividers
    """
    chunks = create_chunks_from_paragraphs(text, max_chunk_size=max_chunk_size)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    dict = initialize_key_value_summary()
    keys = list(dict.keys())
    summaries = []
    
    generation_config = GenerationConfig(
        max_length=model_max_tokens,
        min_length=3,
        length_penalty=1.0,
        num_beams=8,
        do_sample=True,
        temperature=0.9,
        top_k=40,
        top_p=0.9,
    )
    
    for chunk in chunks:
        prompt = f'Using this list: {keys}, summarize this note: {chunk}'
        inputs = tokenizer(chunk, return_tensors="pt").to(device)
        # , truncation=True, max_length=model_max_tokens
        
        if inputs["input_ids"].shape[1] > model_max_tokens:
            print(f"WARNING: Chunk exceeded {model_max_tokens} tokens, truncating.")
            
        summary_ids = model.generate(
            inputs["input_ids"], 
            generation_config=generation_config
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        # print(f"Generated summary: {summary}")
        summaries.append(summary)
        

    final_summary = "\n----------------------------------------------------------------------------------------------------\n".join(summaries)
    return final_summary

def process_notes(notes_folder, output_folder, model, tokenizer):
    """
    Processes clinical notes for each patient and generates summaries.

    Args:
        notes_folder (str): Path to folder containing patient notes (PDF/DOCX)
        output_folder (str): Path to save generated summaries
        model (BartForConditionalGeneration): Pre-trained BART model
        tokenizer (BartTokenizer): Tokenizer for the BART model
    """
    patient_files = {}
    
    for file_name in os.listdir(notes_folder):
        if not (file_name.endswith(".pdf") or file_name.endswith(".docx")):
            continue
        patient_id = file_name.split("_")[0]
        if patient_id not in patient_files:
            patient_files[patient_id] = []
        patient_files[patient_id].append(file_name)

    for patient_id, files in patient_files.items():
        print(f"Processing patient {patient_id}...")
        combined_text = ""
        for file_name in files:
            file_path = os.path.join(notes_folder, file_name)
            if file_name.endswith(".pdf"):
                combined_text += extract_text_from_pdf(file_path)
            elif file_name.endswith(".docx"):
                combined_text += extract_text_from_word(file_path)

        summary = generate_combined_summary(model, tokenizer, combined_text)
        output_file_path = os.path.join(output_folder, f"{patient_id}_summary.txt")
        
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            print(f"Saved summary to {output_file_path}")
            output_file.write(f"{summary}\n")

def main(notes_folder, output_folder):
    """
    Main function to load model and process clinical notes.

    Args:
        notes_folder (str): Path to folder containing clinical notes
        output_folder (str): Path to save generated summaries
    """
    model, tokenizer = load_model_and_tokenizer()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    process_notes(notes_folder, output_folder, model, tokenizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize clinical notes using BART based on specific criteria")
    parser.add_argument('--notes_folder', type=str, required=True, help="Path to the folder containing clinical notes")
    parser.add_argument('--output_folder', type=str, required=True, help="Path to the folder to save the summaries")
    
    args = parser.parse_args()
    main(args.notes_folder, args.output_folder)
