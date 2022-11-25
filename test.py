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

from torchvision.models import vgg19, VGG19_Weights

from PIL import Image


# m = nn.MaxPool1d(2, stride=2)
# input = torch.randn(25088)
# print(input.shape)
# output = m(input)
# print(output.shape)


image = Image.open('./data/news_images/real/politifact_1014.jpg')


# initialize the weight transform
weigths = VGG19_Weights.DEFAULT
preprocess = weigths.transforms()
# apply the transform to the image
image_transformed = preprocess(image) if image else None
# initialize the model
model = vgg19(weights=weigths)
# select the layer to extract features from
layer = model._modules.get('avgpool')
# set model to evaluation mode
model.eval()
# create empty embedding
embedding = torch.zeros(25088)
if image_transformed is not None:
    # create a function that will copy the output of a layer
    def copy_data(m, i, o):
        embedding.copy_(o.flatten())
    # attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # run the model on our transformed image
    model(image_transformed.unsqueeze(0))
    # detach our copy function from the layer
    h.remove()
# perform max pooling on the embedding to achieve a fixed size of 200
print('Squeeze embedding: ', embedding.squeeze(0).shape)
print('Unsqueeze embedding: ', embedding.unsqueeze(0).shape)
embedding = F.max_pool1d(embedding.unsqueeze(0), 125).squeeze(0)
# return the feature vector
print(embedding.size())
