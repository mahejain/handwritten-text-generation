import requests
import os

def download_font():
    url = "https://github.com/google/fonts/raw/main/ofl/caveat/Caveat-Regular.ttf"
    
    font_path = "handwriting.ttf"
    
    if os.path.exists(font_path):
        print("Font already downloaded!")
        return
    
    print("Downloading handwriting font...")
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(font_path, "wb") as f:
            f.write(response.content)
        print("Font downloaded: handwriting.ttf")
    else:
        print("Download failed, try again!")

if __name__ == "__main__":
    download_font()