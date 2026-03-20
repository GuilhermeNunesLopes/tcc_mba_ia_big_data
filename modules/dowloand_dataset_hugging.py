import os

from huggingface_hub import hf_hub_download, login
import lzma
import shutil

#Define the repository ID and filename on Hugging Face
def download_and_decompress(repo_id, filename, local_dir="logs_hugging"):
    # 0. Log in to Hugging Face (if not already logged in)   
    login()
    # 1. Download the file from Hugging Face
    file_path = hf_hub_download(
        repo_id=repo_id, 
        filename=filename, 
        repo_type="dataset", 
        local_dir=local_dir
        )
    
    os.listdir("logs_hugging")  # Check the contents of the directory
    print (f"Downloaded file path: {file_path}")  # Print the path of the downloaded file
    # Check file size and first bytes
    file_size = os.path.getsize(file_path)
    print(f"File size: {file_size}")


    return file_path