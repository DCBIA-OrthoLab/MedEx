import re
import csv
import fitz
import string
from docx import Document
from transformers import BartTokenizer, BartForConditionalGeneration

def load_model_and_tokenizer():
    """
    Load the BART model and tokenizer from the Hugging Face Transformers library.

    Returns:
        BartTokenizer, BartForConditionalGeneration: The tokenizer and model objects.
    """
    model_name_or_path = "/path/to/your/model"
    tokenizer = BartTokenizer.from_pretrained(model_name_or_path)
    model = BartForConditionalGeneration.from_pretrained(model_name_or_path)
    return model, tokenizer

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file using the PyMuPDF library (fitz).

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Cleaned text extracted from the PDF.
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return clean_text(text)

def extract_text_from_word(docx_path):
    """
    Extracts text from a Word document using the python-docx library.

    Args:
        docx_path (str): Path to the Word document.

    Returns:
        str: Cleaned text extracted from the Word document.
    """
    doc = Document(docx_path)
    return "\n".join([clean_text(paragraph.text) for paragraph in doc.paragraphs])

def clean_text(text):
    """
    Cleans the text by replacing specific characters with their desired replacements.
    
    Args:
        text (str): The input text to clean.
    
    Returns:
        str: The cleaned text.
    """
    replacements = {
        "’": "'",
        "–": "-"
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def split_text_by_paragraphs(text):
    """
    Splits input text into meaningful paragraphs while preserving list items and section headers.

    Args:
        text (str): The input text to be split into paragraphs

    Returns:
        list: A list of strings where each string represents a logical paragraph
    """
    # Normalize line breaks
    normalized_text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Define common section headers that should act as paragraph boundaries
    section_headers = ["CLINICAL EXAMINATION", "RADIOGRAPHIC EVALUATION"]
    header_pattern = r'(' + '|'.join(section_headers) + r')\.?'

    # Split based on double newlines, numbered lists, or section headers
    paragraphs = re.split(r'\n\s*\n|\n(?=\d+\.\s)|\n(?=\-)|\n(?=\*)|' + header_pattern, normalized_text)

    merged_paragraphs = []
    current_paragraph = ""

    for para in paragraphs:
        if para is None:
            continue

        para = para.strip()  # Remove leading and trailing whitespace

        if para in section_headers:
            # Treat section header as a separate paragraph
            if current_paragraph:
                merged_paragraphs.append(current_paragraph.strip())
            current_paragraph = para  # Start new paragraph with the section header
        elif re.match(r'^\d+\.\s|^[\-*]\s', para) or (current_paragraph and len(current_paragraph) < 150):
            # Add list items to the current paragraph
            current_paragraph += "\n" + para
        else:
            # Append current paragraph if it's not empty and reset for the new paragraph
            if current_paragraph:
                merged_paragraphs.append(current_paragraph.strip())
            current_paragraph = para  # Start a new paragraph

    # Add any remaining text as the last paragraph
    if current_paragraph:
        merged_paragraphs.append(current_paragraph.strip())

    return merged_paragraphs

def create_chunks_from_paragraphs(text, max_chunk_size=1800):
    """
    Creates content chunks from paragraphs while respecting section boundaries and size limits.

    Args:
        text (str): The input text to be chunked
        max_chunk_size (int, optional): Maximum allowed character length per chunk. Defaults to 1800.

    Returns:
        list: A list of text chunks where each chunk is under the max_chunk_size limit
    """
    section_headers = ["CLINICAL EXAMINATION", "CLINICAL EVALUATION", "RADIOGRAPHIC EXAMINATION", "RADIOGRAPHIC EVALUATION"]

    def split_to_sentences(paragraph, max_size):
        """
        Splits a paragraph into sentence chunks that each fit within the specified maximum size.
    
        Args:
            paragraph (str): The input paragraph text to split
            max_size (int): Maximum allowed character length for each output chunk
    
        Returns:
            list: A list of sentence chunks where each chunk's length <= max_size
        """
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)  # Split on sentence boundaries
        chunk = ""
        chunks = []

        for sentence in sentences:
            if len(chunk) + len(sentence) + 1 <= max_size:
                chunk += sentence + " "
            else:
                if chunk:
                    chunks.append(chunk.strip())
                chunk = sentence + " "

        if chunk:
            chunks.append(chunk.strip())

        return chunks

    paragraphs = split_text_by_paragraphs(text)
    chunks = []
    current_chunk = ""

    i = 0
    while i < len(paragraphs):
        para = re.sub(r'\s{2,}', ' ', paragraphs[i])
        # Check if this paragraph starts with a section header
        if any(para.lower().startswith(header.lower()) for header in section_headers):
            # Accumulate the entire section: header + following non-header paragraphs.
            section_paragraphs = [para]
            j = i + 1
            while j < len(paragraphs):
                next_para = re.sub(r'\s{2,}', ' ', paragraphs[j])
                if any(next_para.lower().startswith(header.lower()) for header in section_headers):
                    break
                section_paragraphs.append(next_para)
                j += 1
            section_text = "\n\n".join(section_paragraphs)
            
            # If current chunk plus the whole section fits, add it.
            if len(current_chunk) + len(section_text) + 1 <= max_chunk_size:
                current_chunk += section_text + "\n\n"
            else:
                # Flush current_chunk if not empty.
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                # If the section itself is small enough, start a new chunk with it.
                if len(section_text) <= max_chunk_size:
                    current_chunk = section_text + "\n\n"
                else:
                    # Otherwise, process the section piece by piece.
                    for sec_para in section_paragraphs:
                        sec_para = re.sub(r'\s{2,}', ' ', sec_para)
                        if len(sec_para) <= max_chunk_size:
                            if len(current_chunk) + len(sec_para) + 1 <= max_chunk_size:
                                current_chunk += sec_para + "\n\n"
                            else:
                                if current_chunk:
                                    chunks.append(current_chunk.strip())
                                current_chunk = sec_para + "\n\n"
                        else:
                            # Split long paragraphs by sentences.
                            para_sentences = split_to_sentences(sec_para, max_chunk_size)
                            for sentence_chunk in para_sentences:
                                if len(current_chunk) + len(sentence_chunk) + 1 <= max_chunk_size:
                                    current_chunk += sentence_chunk + " "
                                else:
                                    if current_chunk:
                                        chunks.append(current_chunk.strip())
                                    current_chunk = sentence_chunk + " "
            i = j  # Move past the entire section.
        else:
            # Regular paragraph (non-header)
            if len(current_chunk) + len(para) + 1 <= max_chunk_size:
                current_chunk += para + "\n\n"
            else:
                # If the paragraph itself is too long, split it.
                para_sentences = split_to_sentences(para, max_chunk_size)
                for sentence_chunk in para_sentences:
                    if len(current_chunk) + len(sentence_chunk) + 1 <= max_chunk_size:
                        current_chunk += sentence_chunk + " "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence_chunk + " "
            i += 1


    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
    
def initialize_key_value_summary():
    """
    Initialize a dictionary with default values based on the expected types.

    Returns:
        dict: Dictionary with default values assigned based on the types.
    """
    KEYS_AND_TYPES = {
    "patient_id": str,
    "patient_age": str,
    "headache_intensity": str,
    "headache_frequency": str,
    "headache_location": str,
    "migraine_history": str,
    "migraine_frequency": str,
    "average_daily_pain_intensity": str,
    "diet_score": str,
    "tmj_pain_rating": str,
    "disability_rating": str,
    "jaw_function_score": str,
    "jaw_clicking": str,
    "jaw_crepitus": str,
    "jaw_locking": str,
    "maximum_opening": str,
    "maximum_opening_without_pain": str,
    "disc_displacement": str,
    "muscle_pain_score": str,
    "muscle_spasm_present": str,
    "muscle_tenderness_present": str,
    "muscle_stiffness_present": str,
    "muscle_soreness_present": str,
    "joint_pain_areas": str,
    "joint_arthritis_present": str,
    "neck_pain_present": str,
    "back_pain_present": str,
    "earache_present": str,
    "tinnitus_present": str,
    "vertigo_present": str,
    "hearing_loss_present": str,
    "hearing_sensitivity_present": str,
    "sleep_apnea_diagnosed": str,
    "sleep_disorder_type": str,
    "airway_obstruction_present": str,
    "anxiety_present": str,
    "depression_present": str,
    "stress_present": str,
    "autoimmune_condition": str,
    "fibromyalgia_present": str,
    "chronic_fatigue_present": str,
    "current_medications": str,
    "previous_medications": str,
    "adverse_reactions": str,
    "appliance_history": str,
    "current_appliance": str,
    "cpap_used": str,
    "apap_used": str,
    "bipap_used": str,
    "physical_therapy_status": str,
    "pain_onset_date": str,
    "pain_duration": str,
    "pain_frequency": str,
    "onset_triggers": str,
    "pain_relieving_factors": str,
    "pain_aggravating_factors": str
    }
    
    defaults = {
        str: "",
    }
    return {key: defaults[expected_type] for key, expected_type in KEYS_AND_TYPES.items()}
