# %% 
import os
import ndjson
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
from bokeh.palettes import Spectral10
from bokeh.resources import INLINE
import bokeh.io
from io import BytesIO
import io
import base64
from tqdm import tqdm

bokeh.io.output_notebook(INLINE)

# %% 
with open('../0519_2000_embeddings_2d.ndjson', 'r') as fin:
    embeddings_2d = ndjson.load(fin)

# %%
meta = pd.read_csv('../data/analysis_subset/subset_metadata.csv')

filenames = [embeddings_2d[x]['file'] for x in range(len(embeddings_2d))]


ids = []
for filename in filenames:
    ids.append(filename.split('_at_iteration_1000.png')[0])
orig_img_filenames = [f"{id}.jpg" for id in ids]


meta_full = pd.merge(meta, pd.DataFrame({'style_image_filename': filenames, "orig_image_filename": orig_img_filenames, "ID": ids}), on="ID", how='outer')

x = [embeddings_2d[x]['embedding'][0] for x in range(len(embeddings_2d))]
y = [embeddings_2d[y]['embedding'][1] for y in range(len(embeddings_2d))]

meta_full["embedding_x"] = x
meta_full["embedding_y"] = y

style_images = [f"../img/{filenames[i]}" for i in range(len(embeddings_2d))]
orig_images = [f"../data/analysis_subset/img/{i}" for i in meta_full['orig_image_filename']]
artists = list(meta_full['artist'])



# %%
# unique_artist = list(set(artists))
# id = []
# period = []
# for artist in unique_artist:
#     artist_sub = meta_full[meta_full['artist']== artist] 
#     mid = np.mean(artist_sub['dating_clean']) #max(artist_sub['dating_clean']) - min(artist_sub['dating_clean']) + min(artist_sub['dating_clean'])
#     for i, dat in enumerate(artist_sub['dating_clean']):
#         id.append(list(artist_sub['ID'])[i])
#         if dat >= mid:
#             period.append(f"{artist}_late")
#         else:
#             period.append(f"{artist}_early")

# meta_fuller = pd.merge(meta_full, pd.DataFrame({'ID': id, "artist_split": period}), on="ID", how='outer')






# %% 
def np_image_to_base64(im_matrix):
    im = Image.fromarray(im_matrix)
    buffer = io.BytesIO()
    im.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url

# %%
image_arrays = []
for img in tqdm(orig_images):
    an_image = Image.open(img).resize((200, 150), Image.BICUBIC)
    image_arrays.append(np.asarray(an_image).astype(np.uint8))

# %%
df = pd.DataFrame({
        'x': x,
        'y': y,
        'file': filenames,
        'image': list(map(np_image_to_base64, image_arrays)),
        "artist": artists
        })

datasource = ColumnDataSource(df)

color_mapping = CategoricalColorMapper(factors=[art for art in set(list(meta_full['artist']))], palette=["red", "blue", "green", "orange", "pink", "purple", "brown", "black"])

# %%
plot_figure = figure(
    title='UMAP projection of the Digits dataset',
    plot_width=1000,
    plot_height=1000,
    tools=('pan, wheel_zoom, reset')
)

plot_figure.add_tools(HoverTool(tooltips="""
<div>
    <div>
        <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/>
    </div>
    <div>
        <span style='font-size: 12px'>@artist</span>
    </div>
</div>
"""))

plot_figure.scatter(
    'x',
    'y',
    source=datasource,
    color=dict(field="artist", transform=color_mapping),
    line_alpha=0.6,
    fill_alpha=0.6,
    size=4
)

show(plot_figure)


# %% 
''' Produce sprite image
'''
# artist = "0520_monet"
# images = [Image.open(f"../img/{filename}").resize((300,300)) for filename in filenames]
# image_width, image_height = images[0].size
# one_square_size = int(np.ceil(np.sqrt(len(images))))
# master_width = (image_width * one_square_size) 
# master_height = image_height * one_square_size
# spriteimage = Image.new(
#     mode='RGBA',
#     size=(master_width, master_height),
#     color=(0,0,0,0))  # fully transparent
# for count, image in enumerate(images):
#     div, mod = divmod(count,one_square_size)
#     h_loc = image_width*div
#     w_loc = image_width*mod    
#     spriteimage.paste(image,(w_loc,h_loc))
# spriteimage.convert("RGB").save(f'../plots/{artist}_sprite.jpg', transparency=0)



# %%
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram



def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

# %%
embeddings = [embeddings_2d[x]['embedding'] for x in range(len(embeddings_2d))]

# %%

clustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(embeddings)

fig = plt.figure(figsize=(20, 10))
plt.title("Hierachical Clustering Dendogram (NMF)")
# plot the top three levels of the dendrogram
plot_dendrogram(clustering, truncate_mode='level', p=5, labels=clustering.labels_)
plt.xlabel("Number of points in cluster (or index of point if no parenthesis).")
#fig.savefig("20211130_hierachical_clustering_nmf.png")
# %%
