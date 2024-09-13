import fitz
import os

# Convert PDF files to txt by reading each page and extracting its text content
def convert_pdf_to_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Directories for PDF input files and txt output files
pdf_directory = "./papers/"
text_directory = "./texts/"
os.makedirs(text_directory, exist_ok=True)

# Get list of all PDF files in input directory
pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

# Convert each PDF file to txt and save to output directory
for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_directory, pdf_file)
    text = convert_pdf_to_text(pdf_path)
    text_path = os.path.join(text_directory, pdf_file.replace('.pdf', '.txt'))
    with open(text_path, 'w') as text_file:
        text_file.write(text)
