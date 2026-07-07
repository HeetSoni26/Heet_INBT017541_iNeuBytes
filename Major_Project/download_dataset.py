# download_dataset.py
import os
import zipfile
import urllib.request

def download_dataset():
    dataset_dir = 'data'
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    zip_path = os.path.join(dataset_dir, 'cornell_movie_dialogs.zip')
    
    if not os.path.exists(zip_path):
        print("Downloading Cornell Movie-Dialogs Corpus (~84MB)...")
        url = "http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.")
    
    if not os.path.exists(os.path.join(dataset_dir, 'cornell_movie_dialogs_corpus')):
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
        print("Extraction complete.")
        
if __name__ == "__main__":
    download_dataset()
