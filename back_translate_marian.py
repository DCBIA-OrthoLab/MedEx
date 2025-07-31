from transformers import MarianMTModel, MarianTokenizer
import torch
import re
import argparse
import os
import glob

def split_sentences(text):
    """Simple sentence splitter that handles common cases"""
    # Split at sentence-ending punctuation followed by whitespace and capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    # If no splits found, split by max length
    if len(sentences) == 1 and len(sentences[0]) > 100:
        return re.split(r'(?<=\S)\s+', text, maxsplit=len(text)//100)
    
    return sentences

def translate_text(text, model, tokenizer, max_length=512):
    """Translate text using MarianMT model"""
    device = model.device
    model.eval()
    
    # Tokenize input
    inputs = tokenizer(
        [text], 
        return_tensors="pt", 
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)
    
    # Generate translation
    translated = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=5,
        early_stopping=True
    )
    
    # Decode output
    return tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

def back_translate(text, forward_model, forward_tokenizer, backward_model, backward_tokenizer, max_chunk_length=400):
    """
    Perform back-translation with MarianMT models
    
    Args:
        text: Input text string
        forward_model: Model for source → pivot translation
        forward_tokenizer: Tokenizer for forward model
        backward_model: Model for pivot → source translation
        backward_tokenizer: Tokenizer for backward model
        max_chunk_length: Token limit per chunk
    
    Returns:
        pivot_text: Intermediate translation to pivot language
        back_text: Back-translated text to source language
    """
    # Split text into sentences
    sentences = split_sentences(text)
    pivot_sentences = []
    back_sentences = []
    
    for i, sentence in enumerate(sentences):
        # Skip empty sentences
        if not sentence.strip():
            pivot_sentences.append("")
            back_sentences.append("")
            continue
            
        # Split long sentences into chunks
        inputs = forward_tokenizer(sentence, return_tensors="pt")["input_ids"][0]
        chunk_indices = torch.split(inputs, max_chunk_length)
        
        pivot_parts = []
        back_parts = []
        for chunk_ids in chunk_indices:
            if chunk_ids.numel() == 0:
                continue
                
            chunk_text = forward_tokenizer.decode(chunk_ids, skip_special_tokens=True)
            
            # Translate to pivot language
            pivot_text = translate_text(
                chunk_text, 
                forward_model, 
                forward_tokenizer
            )
            
            # Translate back to source language
            back_translated = translate_text(
                pivot_text, 
                backward_model, 
                backward_tokenizer
            )
            
            pivot_parts.append(pivot_text)
            back_parts.append(back_translated)
        
        # Combine parts for the sentence
        pivot_sentence = " ".join(pivot_parts)
        back_sentence = " ".join(back_parts)
        
        # Add period at end of sentence if missing
        if not back_sentence.endswith(('.', '!', '?')) and i < len(sentences) - 1:
            back_sentence += '.'
        if not pivot_sentence.endswith(('.', '!', '?')) and i < len(sentences) - 1:
            pivot_sentence += '.'
            
        pivot_sentences.append(pivot_sentence)
        back_sentences.append(back_sentence)
    
    return " ".join(pivot_sentences), " ".join(back_sentences)

def process_files(input_folder, output_folder, src_lang="en", pivot_lang="fr", save_pivot=True):
    """Process all patient summary files in input folder"""
    # Create output folder if needed
    os.makedirs(output_folder, exist_ok=True)
    pivot_folder = os.path.join(output_folder, "pivot") if save_pivot else None
    if pivot_folder:
        os.makedirs(pivot_folder, exist_ok=True)
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model names for MarianMT (Helsinki-NLP models)
    forward_model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{pivot_lang}"
    backward_model_name = f"Helsinki-NLP/opus-mt-{pivot_lang}-{src_lang}"
    
    print(f"Loading forward model: {forward_model_name}")
    print(f"Loading backward model: {backward_model_name}")
    
    # Load models and tokenizers
    forward_tokenizer = MarianTokenizer.from_pretrained(forward_model_name)
    forward_model = MarianMTModel.from_pretrained(forward_model_name).to(device)
    
    backward_tokenizer = MarianTokenizer.from_pretrained(backward_model_name)
    backward_model = MarianMTModel.from_pretrained(backward_model_name).to(device)
    
    # Find all patient summary files
    input_files = glob.glob(os.path.join(input_folder, "*.txt"))
    
    print(f"Found {len(input_files)} patient files to process")
    print(f"Saving pivot translations: {save_pivot}")
    
    for file_path in input_files:
        # Get patient ID from filename
        filename = os.path.basename(file_path)
        patient_id = filename.split('_')[0]
        output_path = os.path.join(output_folder, f"{patient_id}_backtranslated.txt")
        pivot_path = os.path.join(pivot_folder, f"{patient_id}_pivot.txt") if save_pivot else None
        
        print(f"Processing patient {patient_id}...")
        
        # Read input file line by line to preserve structure
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Process each line individually to preserve newlines
        pivot_lines = []
        back_translated_lines = []
        
        for i, line in enumerate(lines):
            # Preserve empty lines exactly
            if not line.strip():
                pivot_lines.append("\n")
                back_translated_lines.append("\n")
                continue
                
            # Process non-empty lines
            try:
                # Preserve leading/trailing whitespace
                leading_ws = len(line) - len(line.lstrip())
                trailing_ws = len(line) - len(line.rstrip())
                content = line.strip()
                
                # Only process if there's actual content
                if content:
                    pivot_content, back_content = back_translate(
                        text=content,
                        forward_model=forward_model,
                        forward_tokenizer=forward_tokenizer,
                        backward_model=backward_model,
                        backward_tokenizer=backward_tokenizer
                    )
                    # Reconstruct line with original whitespace
                    back_line = (' ' * leading_ws) + back_content + (' ' * trailing_ws)
                    pivot_line = (' ' * leading_ws) + pivot_content + (' ' * trailing_ws)
                else:
                    back_line = line
                    pivot_line = line
                    
            except Exception as e:
                print(f"Error processing line {i+1} for {patient_id}: {str(e)}")
                back_line = line
                pivot_line = line
                
            back_translated_lines.append(back_line)
            pivot_lines.append(pivot_line)
        
        # Save back-translated output
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(back_translated_lines)
        
        # Save pivot translation if requested
        if save_pivot and pivot_path:
            with open(pivot_path, 'w', encoding='utf-8') as f:
                f.writelines(pivot_lines)
        
        print(f"  Saved back-translation for {patient_id}")
        if save_pivot:
            print(f"  Saved pivot translation for {patient_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Back-translate patient summaries using MarianMT models')
    parser.add_argument('--input', type=str, required=True, 
                        help='Input folder containing patient summary files')
    parser.add_argument('--output', type=str, required=True,
                        help='Output folder for back-translated files')
    parser.add_argument('--src_lang', type=str, default='en',
                        help='Source language code (default: en)')
    parser.add_argument('--pivot_lang', type=str, default='fr',
                        help='Pivot language code (default: fr)')
    parser.add_argument('--save_pivot', action='store_true',
                        help='Save intermediate pivot language translations')
    
    args = parser.parse_args()
    
    print(f"Starting back-translation from {args.src_lang} via {args.pivot_lang}")
    print(f"Input folder: {args.input}")
    print(f"Output folder: {args.output}")
    print(f"Save pivot translations: {args.save_pivot}")
    
    process_files(
        input_folder=args.input,
        output_folder=args.output,
        src_lang=args.src_lang,
        pivot_lang=args.pivot_lang,
        save_pivot=args.save_pivot
    )
    
    print("Back-translation completed!")