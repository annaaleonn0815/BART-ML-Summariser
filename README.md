# Tool for Summarising Machine Learning Research Papers

This repository serves as a demonstration for the implementation and findings of my dissertation project,"Enhancing Summarisation of Machine Learning Literature using Fine-Tuned BART: A Transfer Learning Approach", which addresses the challenge of effectively summarising machine learning engineering literature, which is characterised by dense technical content and specialised terminology. The goal is to enhance the summarisation capabilities of Natural Language Processing (NLP) models to process and summarise these complex texts.

To achieve this, I used the **BART model**, a sequence-to-sequence transformer, fine-tuned via transfer learning on a specialised dataset of machine learning papers. The key sections of research papers (introduction, results, and conclusion) were selected for training the model, and its performance was evaluated using ROUGE scores. The fine-tuned model showed a significant improvement in summarisation quality, achieving a 48% increase in F1 score over the pre-trained version.

A **Graphical User Interface (GUI)** was developed to provide an accessible interface for users to interact with the model, enabling them to load input files, adjust summary length, and save generated summaries. This tool has been designed to assist researchers in quickly generating concise and relevant summaries of complex academic papers.

## Features

- **Fine-Tuned BART Model**: The BART model was fine-tuned using a custom dataset of machine learning papers to improve its ability to handle technical language and complex document structures.
- **Graphical User Interface (GUI)**: A Python-based GUI (built using Tkinter) that enables users to:
  - Load research papers in `.txt` format.
  - Set the desired summary length.
  - Generate summaries and display them within the application.
  - Save the generated summaries to a file.
- **Error Handling**: The GUI provides robust error-handling, ensuring smooth operation even when missing or incorrect files are loaded.

## Installation


Ensure you have the following installed:

- Python 3.7 or higher
- The required Python libraries (see `requirements.txt`)

## Usage

1. Clone the repository to your local machine.
2. Activate your virtual environment.
3. Start the application by running the following command: ```python graphical_interface.py```
4. Select "Load Text File" and navigate to the ```GUI_documents``` folder. Sample research papers are included in the folder.
5. Select a paper and click "Open".
6. Adjust the summary length as desired and click the "Generate Summary" button.
7. Save the generated summary if needed.



### Please note that there are a couple of files that were unable to upload to Github due to their size; for the complete code please contact me at al22581@essex.ac.uk 
