# %%
import os
from tqdm import tqdm
import ndjson

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import vgg19



def preprocess_image(image_path, img_nrows, img_ncols):
    # Util function to open, resize and format pictures into appropriate tensors
    img = keras.preprocessing.image.load_img(
        image_path, target_size=(img_nrows, img_ncols)
    )
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)


# Build a VGG19 model loaded with pre-trained ImageNet weights
model = vgg19.VGG19(weights="imagenet", include_top=False)

# Get the symbolic outputs of each "key" layer (we gave them unique names).
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

# Set up a model that returns the activation values for every layer in VGG19 (as a dict).
feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)


# Image
image_path = "../img/17539_at_iteration_1000.png"

# Dimensions of the image
width, height = keras.preprocessing.image.load_img(image_path).size
img_nrows = 400
img_ncols = int(width * img_nrows / height)

# Apply to image
img = preprocess_image(image_path, img_nrows, img_ncols)

# Extract features
features = feature_extractor(img)

# Last layer + flatten
last_layer_flat = layers.Flatten()(features['block5_pool']).numpy()

# %%
def extract_embedding(path):
    # dimensions
    width, height = keras.preprocessing.image.load_img(image_path).size
    img_nrows = 400
    img_ncols = int(width * img_nrows / height)

    # preprocess
    img = preprocess_image(image_path, img_nrows, img_ncols)

    # extract features
    features = feature_extractor(img)

    # last pooling layer + flatten
    last_layer_flat = layers.Flatten()(features['block5_pool']).numpy()
    return last_layer_flat


# %%
# go it for all
img_paths = [path for path in os.listdir('../img') if path.endswith('.png')]

embeddings = []
for path in tqdm(img_paths):
    flat_layer = extract_embedding(path)
    embeddings.append(flat_layer)

embeddings = np.vstack(embeddings)


# %%
import umap

reducer = umap.UMAP(random_state=42, metric='cosine', n_epochs=2000, n_neighbours=10)
embeddings_2d = reducer.fit_transform(embeddings)


# %%
embeddings_out = embeddings_2d.tolist()

out = []
for emb, fn in zip(embeddings_out, img_paths):
    out.append({
        'file': fn,
        'embedding': emb
    })

with open('../0520_1700_embeddings_2d.ndjson', 'w') as fout:
    ndjson.dump(out, fout)