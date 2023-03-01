# from PIL import Image
# import requests
# from io import BytesIO

# import torch
# from torchvision.models import vgg19, VGG19_Weights

# # load image
# response = requests.get('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS80DJ8tkbXY3gqhVvdcQziIMS0hbbe9pP2wIReQ1AcEI-YXM11Jh76NY00qPydZW4vUhQ&usqp=CAU')
# img1 = Image.open(BytesIO(response.content))

# response = requests.get('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTPqPBZ3QRTQ_WcTqBXJog2yXp3fpWD_nQNbwLZlxzeVEjCqoe11KfiHO7faU0ws5EP62E&usqp=CAU')
# img2 = Image.open(BytesIO(response.content))

# def get_image_features(image):
#     # initialize the weight transform
#     weigths = VGG19_Weights.DEFAULT
#     preprocess = weigths.transforms()
#     # apply the transform to the image
#     image_transformed = preprocess(image)
#     # initialize the model
#     model = vgg19(weights=weigths)
#     # select the layer to extract features from
#     layer = model._modules.get('avgpool')
#     # set model to evaluation mode
#     model.eval()
#     # create empty embedding
#     embedding = torch.zeros(25088)
#     # create a function that will copy the output of a layer
#     def copy_data(m, i, o):
#         embedding.copy_(o.flatten())
#     # attach that function to our selected layer
#     h = layer.register_forward_hook(copy_data)
#     # run the model on our transformed image
#     model(image_transformed.unsqueeze(0))
#     # detach our copy function from the layer
#     h.remove()
#     # return the feature vector
#     return embedding


# embed1 = get_image_features(img1)
# embed2 = get_image_features(img2)

# print(embed1.shape)
# print(embed2.shape)

# # calculate cosine similarity
# cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
# cos_sim = cos(embed1, embed2)
# print(cos_sim)

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torchvision.models import vgg19, VGG19_Weights

from PIL import Image


# m = nn.MaxPool1d(2, stride=2)
# input = torch.randn(25088)
# print(input.shape)
# output = m(input)
# print(output.shape)


#image = Image.open('./data/news_images/real/politifact_1014.jpg')


# # initialize the weight transform
# weigths = VGG19_Weights.DEFAULT
# preprocess = weigths.transforms()
# # apply the transform to the image
# image_transformed = preprocess(image) if image else None
# # initialize the model
# model = vgg19(weights=weigths)
# # select the layer to extract features from
# layer = model._modules.get('avgpool')
# # set model to evaluation mode
# model.eval()
# # create empty embedding
# embedding = torch.zeros(25088)
# if image_transformed is not None:
#     # create a function that will copy the output of a layer
#     def copy_data(m, i, o):
#         embedding.copy_(o.flatten())
#     # attach that function to our selected layer
#     h = layer.register_forward_hook(copy_data)
#     # run the model on our transformed image
#     model(image_transformed.unsqueeze(0))
#     # detach our copy function from the layer
#     h.remove()
# # perform max pooling on the embedding to achieve a fixed size of 200
# print('Squeeze embedding: ', embedding.squeeze(0).shape)
# print('Unsqueeze embedding: ', embedding.unsqueeze(0).shape)
# embedding = F.max_pool1d(embedding.unsqueeze(0), 125).squeeze(0)
# # return the feature vector
# print(embedding.size())


# url encode string and print
# import urllib.parse

# url = 'http://web.archive.org/web/20180416210645im_/http://sciencevibe.com/wp-content/uploads/2018/04/78-Year-Old-CIA-Agent-Confesses-On-Deathbed-\u201cI-Killed-Marilyn-Monroe\u201d.jpg'

# encoded_url = urllib.parse.quote(url, safe=':/')
# print(encoded_url)


import distance

# top_img_url = "http://web.archive.org/web/20160924061356im_/https://resize.rbl.ms/simage/https%3A%2F%2Fassets.rbl.ms%2F4366383%2F1200x600.jpg/2000%2C2000/vGLtGfaZn2jEFQQR/img.jpg"
# images = [
# "http://web.archive.org/web/20160924061356im_/https://resize.rbl.ms/simage/https%3A%2F%2Fassets.rbl.ms%2F4366383%2F1200x600.jpg/2000%2C2000/vGLtGfaZn2jEFQQR/img.jpg",
#     "http://web.archive.org/web/20160924061356im_/https://d5nxst8fruw4z.cloudfront.net/atrk.gif?account=pUobn1aMp410fn",
#     "http://web.archive.org/web/20160924061356im_/https://assets.rbl.ms/4366383/1200x600.jpg"
# ]

# min_distance = float("inf")
# most_similar_url = ""

# for url in images:
#     curr_distance = distance.levenshtein(top_img_url, url)
#     print(curr)
#     if curr_distance < min_distance and curr_distance != 0:
#         min_distance = curr_distance
#         most_similar_url = url

# print("The most similar URL to the 'top_img' key is:", most_similar_url)


# count total number of image files in data directory with subdirectories fake and real
# import os

# fake_dir = './data/news_images_v2/fake'
# real_dir = './data/news_images_v2/real'

# fake_count = 0
# real_count = 0

# for file in os.listdir(fake_dir):
#     fake_count += 1

# for file in os.listdir(real_dir):
#     real_count += 1

# print("Total number of fake images:", fake_count)
# print("Total number of real images:", real_count)

# from PIL import Image
# import os

# # Open the first image in the folder
# folder_path = './data/news_images_v2/fake'
# images = os.listdir(folder_path)
# first_image = Image.open(os.path.join(folder_path, images[0]))

# # Calculate the size of the final collage image
# num_images = len(images)
# collage_size = (1000, 1000 * num_images)

# # Create a new image with the calculated size
# collage = Image.new('RGB', collage_size)

# # Loop through the images in the folder and paste them onto the collage
# x_offset = 0
# for image_name in images:
#     image_path = os.path.join(folder_path, image_name)
#     image = Image.open(image_path)
#     collage.paste(image, (x_offset, 0))
#     x_offset += image.width

# # Save the final collage image
# collage.save('./data/news_images_v2/collage_fake.jpg')

# folder_path = './data/news_images_v2/real'
# images = os.listdir(folder_path)
# first_image = Image.open(os.path.join(folder_path, images[0]))

# # Calculate the size of the final collage image
# num_images = len(images)
# print(first_image.height)
# collage_size = (1000, 1000 * num_images)

# # Create a new image with the calculated size
# collage = Image.new('RGB', collage_size)

# # Loop through the images in the folder and paste them onto the collage
# x_offset = 0
# for image_name in images:
#     image_path = os.path.join(folder_path, image_name)
#     image = Image.open(image_path)
#     collage.paste(image, (x_offset, 0))
#     x_offset += image.width

# # Save the final collage image
# collage.save('./data/news_images_v2/collage_real.jpg')

#Evan Russenberger-Rosica
#Create a Grid/Matrix of Images

# image = Image.open('./data/news_images_v2/real/politifact_4926.jpg')
# resizer = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
# ])
# transformer = VGG19_Weights.DEFAULT.transforms()
# preprocessed = resizer(image)

# preprocessed.show()


# def make_deepfc(input_size, hidden_sizes, output_size):

#     deepfc = nn.Sequential()
#     deepfc.add_module('input', nn.Linear(input_size, hidden_sizes[0]))
#     deepfc.add_module('relu_0', nn.ReLU())
#     for i in range(len(hidden_sizes) - 1):
#         deepfc.add_module('hidden_{}'.format(i), nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
#         deepfc.add_module('relu_{}'.format(i+1), nn.ReLU())
#     deepfc.add_module('output', nn.Linear(hidden_sizes[-1], output_size))

#     return deepfc


# deepfc = make_deepfc(224*224*3, [100], 200)
# print(deepfc)


import fnmatch
import os
import pandas as pd

dirpath_fake_img = './data/politifact_v2/news_images/fake'
dirpath_real_img = './data/politifact_v2/news_images/real'

dirpath_fake_img_v2 = './data/politifact_v2/news_images_v2/fake'
dirpath_real_img_v2 = './data/politifact_v2/news_images_v2/real'

print('{} / {}'.format(len(os.listdir(dirpath_fake_img_v2)), len(os.listdir(dirpath_fake_img))))
print('{} / {}'.format(len(os.listdir(dirpath_real_img_v2)), len(os.listdir(dirpath_real_img))))

dirpath_fake = './fakenewsnet_dataset_v2/politifact/fake'
dirpath_real = './fakenewsnet_dataset_v2/politifact/real'

print(len(os.listdir(dirpath_fake)))
print(len(os.listdir(dirpath_real)))

df_s = pd.read_csv("./data/politifact_v2_no_ignore_s.tsv", sep="\t")
df_e = pd.read_csv("./data/politifact_v2_no_ignore_en.tsv", sep="\t")
# count numer of rows in politifact dataset
print(len(df_s))
print(len(df_e))
# count number of rows with NAN text
print(df_s['text'].isnull().sum())

