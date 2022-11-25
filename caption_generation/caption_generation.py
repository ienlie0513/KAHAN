import sys
import os
import argparse

from tqdm import tqdm

import pandas as pd
import numpy as np
import torch

from PIL import Image

from lavis.models import load_model_and_preprocess

class CaptionGenerator:
    def __init__(self, model_name, model_type):
        self.model, self.vis_processors, _ = load_model_and_preprocess(name=model_name, model_type=model_type, is_eval=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate_caption(self, image):
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        caption = (self.model.generate({"image": image})[0])
        return caption

def get_images(df, source, dir_path):
    images = []
    for idx in tqdm(range(df.id.shape[0])):
        if df.label[idx] == 1:
            path = dir_path + '/real/' + source + '_' + str(df.id[idx]) + '.jpg'
            if os.path.exists(path):
                images.append({'id': df.id[idx], 'img_source': Image.open(path)})
            else:
                images.append({'id': df.id[idx], 'img_source': None})
        else:
            path = dir_path + '/fake/' + source + '_' + str(df.id[idx]) + '.jpg'
            if os.path.exists(path):
                images.append({'id': df.id[idx], 'img_source': Image.open(path)})
            else:
                images.append({'id': df.id[idx], 'img_source': None})
    return images
    

def get_captions(images, model_name, model_type):
    cap_gen = CaptionGenerator(model_name, model_type)
    captions = []
    for image in tqdm(images):
        if image['img_source'] is not None:
            caption = cap_gen.generate_caption(image['img_source'])
            captions.append({'id': image['id'], 'caption': caption})
            print(type(caption), ": " + caption)
        else:
            captions.append({'id': image['id'], 'caption': None})
    return captions
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", type=str, default="politifact")
    args = parser.parse_args()

    df = pd.read_csv("./data/{}_no_ignore_en.tsv".format(args.platform), sep="\t")
    df['id'] = df['id'].astype(str)
    df = df.fillna('')

    images_dir = "./data/news_images"

    imgs = get_images(df, args.platform, images_dir)
    caption_df = pd.DataFrame(get_captions(imgs, "blip_caption", "base_coco"))

    print(caption_df)
    print(caption_df.columns)

    print(df)
    print(df.columns)

    df = pd.merge(df, caption_df, on='id')
    df.to_csv("./data/{}_no_ignore_en_cap.tsv".format(args.platform), sep = '\t', index=False)


