import pdfplumber
from transformers import BartForConditionalGeneration, BartTokenizer
from langchain.document_loaders import PyPDFLoader
import streamlit as st
import tempfile
import os


# Load pre-trained BART model and tokenizer
model_name = "facebook/bart-large-cnn"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

def load_pdf_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

def summarize_pdfs_from_folder(pdfs_folder):
    summaries = []
    for pdf_file in pdfs_folder:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(pdf_file.read())
        
        # Load and split text from the PDF file
        text = load_pdf_text(temp_path)

        # Generate summary using the BART model
        inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
        summary_ids = model.generate(inputs.input_ids)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

        # Delete the temporary file
        os.remove(temp_path)
    
    return summaries

# Streamlit App
st.title("The Skimmers Multiple PDF Summarizer")

# Allow user to upload PDF files
pdf_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if pdf_files:
    # Generate summaries when the "Generate Summary" button is clicked
    if st.button("Generate Summary"):
        st.write("Summaries:")
        summaries = summarize_pdfs_from_folder(pdf_files)
        for i, summary in enumerate(summaries):
            st.write(f"Summary for PDF {i+1}:")
            st.write(summary)
