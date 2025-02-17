import os
os.environ['HF_HOME'] = "/local/athanasiadisc/cache"
from timeit import default_timer as timer
from sentence_transformers import SentenceTransformer, util
from urllib.request import urlopen
from PIL import Image

from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel

import torch
import numpy as np
import matplotlib.pyplot as plt

# Clip is a multimodal LLMs that leverages the embeddings from different modalities like text and images and Transformer architecure to coccurentrly project them into a 
# common subspace
# In this example we will see how clips works for simple images
# Load an AI-generated image of a puppy playing in the snow
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
image = Image.open(urlopen(puppy_path)).convert("RGB")

caption = "a puppy playing in the snow"
model_id = "openai/clip-vit-base-patch32"

# Load a tokenizer to preprocess the text
clip_tokenizer = CLIPTokenizerFast.from_pretrained(model_id)

# Load a processor to preprocess the images
clip_processor = CLIPProcessor.from_pretrained(model_id)

# Main model for generating text and image embeddings
model = CLIPModel.from_pretrained(model_id)

# Tokenize our input
inputs = clip_tokenizer(caption, return_tensors="pt")
inputs 

# Convert our input back to tokens
clip_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# Create a text embedding
text_embedding = model.get_text_features(**inputs)
text_embedding.shape

# Preprocess image
processed_image = clip_processor(
    text=None, images=image, return_tensors="pt"
)["pixel_values"]

# Prepare image for visualization
img = processed_image.squeeze(0)
img = img.permute(*torch.arange(img.ndim - 1, -1, -1))
img = np.einsum("ijk->jik", img)

# Visualize preprocessed image
plt.imshow(img)
plt.axis("off")
plt.savefig('../../_files/image.png')


# Create the image embedding
image_embedding = model.get_image_features(processed_image)
image_embedding.shape

# Normalize the embeddings
text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
image_embedding /= image_embedding.norm(dim=-1, keepdim=True)

# Calculate their similarity
text_embedding = text_embedding.detach().cpu().numpy()
image_embedding = image_embedding.detach().cpu().numpy()
score = np.dot(text_embedding, image_embedding.T)

# Load SBERT-compatible CLIP model
model = SentenceTransformer("clip-ViT-B-32")

images = image
captions = caption
# Encode the images
image_embeddings = model.encode(images)

# Encode the captions
text_embeddings = model.encode(captions)

#Compute cosine similarities
sim_matrix = util.cos_sim(
    image_embeddings, text_embeddings
)

import pdb
pdb.set_trace() 