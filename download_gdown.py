"""
You can download files from google drive with gdown!
you may have urls in these formats:
https://drive.google.com/file/d/1AZcc77cmDfkWA8f8cs-j-CUuFFQ7tPoK/view
https://drive.usercontent.google.com/download?id=1AZcc77cmDfkWA8f8cs-j-CUuFFQ7tPoK&export=download&authuser=0
Then use the id within the urls to convert it in this format
https://drive.google.com/uc?id=1AZcc77cmDfkWA8f8cs-j-CUuFFQ7tPoK
"""
import gdown

url = 'https://drive.google.com/uc?id=1AZcc77cmDfkWA8f8cs-j-CUuFFQ7tPoK'
output = 'file.tar'
gdown.download(url, output, quiet=False)
