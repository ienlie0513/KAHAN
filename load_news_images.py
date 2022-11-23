import pandas as pd
import os

from PIL import Image
from io import BytesIO
import requests

# make a new directory for the loaded images
dir_path = os.getcwd() + '/data/news_images'
os.makedirs(dir_path, exist_ok=True)
# make a new directory for fake news images
sub_dir_path_fake = os.getcwd() + '/data/news_images/fake'
os.makedirs(sub_dir_path_fake, exist_ok=True)
# make a new directory for real news images
sub_dir_path_real = os.getcwd() + '/data/news_images/real'
os.makedirs(sub_dir_path_real, exist_ok=True)

if __name__ == "__main__":
    df_p = pd.read_csv("./data/politifact_data.tsv", sep="\t")
    df_g = pd.read_csv("./data/gossipcop_data.tsv", sep="\t")

def load_and_store_images(df, source):
    for idx in range(df.id.shape[0]):
        save_path = dir_path + '/real/' + source + '_' + str(df.id[idx]) + '.jpg' if df.label[idx] == 1 else dir_path + '/fake/' + source + '_' + str(df.id[idx]) + '.jpg'
        try:
            if os.path.exists(dir_path + '/' + source + '_' + str(df.id[idx]) + '.jpg'):
                continue
            if isinstance(df.image[idx], str) and df.image[idx] != '':
                if df.image[idx].lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    print("Downloading for: " + str(df.id[idx]))
                    response = requests.get(df.image[idx])
                    img = Image.open(BytesIO(response.content)).convert('RGB')
                    img.save(save_path)
        except Exception as e:
            print("Image not loaded: ", e)

load_and_store_images(df_p, 'politifact')
load_and_store_images(df_g, 'gossipcop')
