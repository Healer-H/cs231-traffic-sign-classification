import os
import gdown
import zipfile

def download_file_from_google_drive(file_id, dest_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest_path, quiet=False)

def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"Extracted {zip_path} to {extract_to}")

def main():
    data_dir = './'
    os.makedirs(data_dir, exist_ok=True)
    
    file_id = '1JcMEKjEH8MKaPX-Aw_Z2-FXTAHMMAVZc' 
    dest_zip_path = os.path.join(data_dir, 'data.zip')
    
    download_file_from_google_drive(file_id, dest_zip_path)
    
    # Giải nén tệp ZIP
    unzip_file(dest_zip_path, data_dir)

    # Xóa tệp ZIP sau khi giải nén
    os.remove(dest_zip_path)
    print(f"Removed {dest_zip_path}")

if __name__ == "__main__":
    main()
