from langchain.vectorstores import FAISS
import os
import logging

from langchain_experimental.text_splitter import SemanticChunker
# from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# ✅ Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")



model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
# text_splitter = SemanticChunker(OpenAIEmbeddings())
hf_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
    )
text_splitter = SemanticChunker(hf_embeddings)

import os

# Define the base directory containing markdown files
BASE_DIR = os.getcwd()+"/data/ubuntu-docs"  # Change this to your folder path

# Function to recursively find and read markdown files
def read_markdown_files(base_dir):
    """
    Recursively reads markdown files from the given directory.
    Returns a list of dictionaries containing folder name, file name, file path, and content.
    """
    markdown_data = []  # List to store file data

    if not os.path.exists(base_dir):
        logging.error(f"Directory not found: {base_dir}")
        return markdown_data

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".md"):  # Check for markdown files
                file_path = os.path.join(root, file)
                folder_name = os.path.basename(root)

                # Read the file content
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    markdown_data.append({
                        "folder": folder_name,
                        "file": file,
                        "path": file_path,
                        "content": content
                    })
                    logging.info(f"Loaded file: {file_path}")
                except Exception as e:
                    logging.error(f"Error reading {file_path}: {e}")
    logging.info(f"Total markdown files loaded: {len(markdown_data)}")

    return markdown_data


def main(text_splitter, markdown_files):
    """
    Splits markdown content into semantic chunks and stores it in a FAISS vector database.
    """
    docs = []
    
    for md in markdown_files:
        try:
            chunks = text_splitter.create_documents([md['content']])
            docs.extend(chunks)
        except Exception as e:
            logging.error(f"Error processing file {md['file']}: {e}")

    if not docs:
        logging.warning("No documents were created. Check if markdown files were loaded properly.")
        return None

    try:
        vectorstore = FAISS.from_documents(docs, hf_embeddings)
        vectorstore.save_local("faiss_index_constitution")
        logging.info("FAISS index saved successfully.")
        return vectorstore
    except Exception as e:
        logging.error(f"Error creating FAISS index: {e}")
        return None

if __name__ == "__main__":
    logging.info("Starting the process...")
    markdown_files = read_markdown_files(BASE_DIR)
    if markdown_files:
        main(text_splitter, markdown_files)
    else:
        logging.warning("No markdown files found. Exiting.")

