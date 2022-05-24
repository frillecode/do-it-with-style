# %%
import re
import ndjson
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import gridspec
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

# %%
# get metadata
meta_data = pd.read_csv('../data/analysis_subset/subset_metadata.csv')
meta_data['ID'] = meta_data['ID'].astype(str)

# load embeddings & extract coordinates in separate columns
with open('../0520_1600_embeddings_2d.ndjson') as fin:
    emb_json = ndjson.load(fin)
    emb = pd.DataFrame(emb_json)
    emb_coordinates = pd.DataFrame(emb['embedding'].tolist(), columns=['X', 'Y'])
    emb = pd.concat([emb, emb_coordinates], axis=1)
    # add id columns
    emb['ID'] = [re.match(r'.*(?=_at_iteration_1000.png)', path).group(0) for path in emb['file'].tolist()]

# merge
df = pd.merge(emb, meta_data, how='left', on='ID')

# %%
# split artists
kupka = df.query('artist == "KUPKA, Frantisek"').sort_values(by='dating_clean')
vangogh = df.query('artist == "GOGH, Vincent van"').sort_values(by='dating_clean')
monet = df.query('artist == "MONET, Claude"').sort_values(by='dating_clean')
goya = df.query('artist == "GOYA Y LUCIENTES, Francisco de"').sort_values(by='dating_clean')


# %%
# 
def find_cluster_prototypes(artist_df, plotting_data=False):
    
    artist_df = artist_df.sort_values(by='dating_clean')
    embeddings = artist_df['embedding'].tolist()
    ids = artist_df['ID'].tolist()

    kmeans_prototypes = []
    _plotting = []
    for k in [1, 2, 3, 4, 5]:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(embeddings)
        inertia = km.inertia_
        centroids = km.cluster_centers_

        coords = []
        for cluster_id, centroid in enumerate(centroids):
            d_centroid = pairwise_distances(
                X=centroid.reshape(1, -1),
                Y=embeddings
            )

            # index of closest document to centroid
            d_centroid = d_centroid[0]
            doc_idx = np.argmin(d_centroid)

            prototype_doc_id = ids[doc_idx]
            prototype_doc_coordinates = embeddings[doc_idx]
            coords.append(prototype_doc_coordinates)

            kmeans_prototypes.append({
                'k': k,
                'inertia': inertia,
                'cluster': cluster_id,
                'prot_id': prototype_doc_id,
                'distance_to_centroid': d_centroid[doc_idx],
                'median_distance_to_centroid': np.median(d_centroid),
                'n_points_in_cluster': len(np.where(km.labels_ == cluster_id)[0])
            })

        # plotting data
        _plotting.append({
            'k': k,
            'labels': km.labels_,
            'centroids': centroids,
            'prototype_coordinates': coords,
            'embeddings': embeddings,
        })

    df_kmeans_prototypes = pd.DataFrame(kmeans_prototypes)

    if plotting_data:
        return df_kmeans_prototypes, _plotting
    else:
        return df_kmeans_prototypes


# %%
kupka_km, kupka_plt = find_cluster_prototypes(kupka, plotting_data=True)

# %%
def plot_prototype_space(plotting_data, k, axis=None):

    k_idx = k - 1

    emb = np.array(plotting_data[k_idx]['embeddings'])
    labels = plotting_data[k_idx]['labels']
    prot_coord = np.array(plotting_data[k_idx]['prototype_coordinates'])
    centroids = plotting_data[k_idx]['centroids']

    if axis:
        # embeddings
        sns.scatterplot(
            x=emb[:, 0],
            y=emb[:, 1],
            hue=labels,
            palette='Set2',
            s=100,
            ax=axis
            )

        # black line connecting centroid & prototype
        for i in range(0, len(centroids)):
            x = [centroids[i][0], prot_coord[i][0]]
            y = [centroids[i][1], prot_coord[i][1]]
            axis.plot(x, y, '-', color='black')
    
        # centroids
        axis.scatter(
            x=centroids[:, 0],
            y=centroids[:, 1],
            color='black',
            # marker='s',
            s=150
        )

        # prototypes
        sns.scatterplot(
            x=prot_coord[:, 0],
            y=prot_coord[:, 1],
            hue=[i for i in range(len(prot_coord))],
            palette='Set2',
            s=250,
            ax=axis
        )

        # get rid of legend
        plt.legend([],[], frameon=False)
        # get rid of ticks
        plt.xticks([])
        plt.yticks([])

    else:
        plt.figure(figsize=(8, 6))
        # embeddings
        sns.scatterplot(
            x=emb[:, 0],
            y=emb[:, 1],
            hue=labels,
            palette='Set2',
            s=100
            )

        # black line connecting centroid & prototype
        for i in range(0, len(centroids)):
            x = [centroids[i][0], prot_coord[i][0]]
            y = [centroids[i][1], prot_coord[i][1]]
            plt.plot(x, y, '-', color='black')
        
        # centroids
        plt.scatter(
            x=centroids[:, 0],
            y=centroids[:, 1],
            color='black',
            # marker='s',
            s=150
        )

        # prototypes
        sns.scatterplot(
            x=prot_coord[:, 0],
            y=prot_coord[:, 1],
            hue=[i for i in range(len(prot_coord))],
            palette='Set2',
            s=250,
        )

        # get rid of legend
        plt.legend([],[], frameon=False)
        # get rid of ticks
        plt.xticks([])
        plt.yticks([])
        # theme
        plt.tight_layout()
        # render
        plt.show()


# %%
# plt
plot_prototype_space(
    kupka_plt,
    5
)

# %%
# load images
def laod_proto_imgs(artist_km):

    img_paths = ['../data/analysis_subset/img/' + img_id + '.jpg' for img_id in artist_km['prot_id'].tolist()]

    paintings = []
    for path in img_paths:
        paintings.append(mpimg.imread(path))
    
    return paintings

# %%
# asd
kupka_paintings = laod_proto_imgs(kupka_km)


fig = plt.figure(figsize=(10, 10), constrained_layout=True) 
gs = gridspec.GridSpec(2, 5, figure=fig) 

# parameters
palette = sns.color_palette('Set2')
edge_width = 5

# top part: clustering
ax0 = plt.subplot(gs[0, :])
plot_prototype_space(kupka_plt, 5, axis=ax0)

# ax1: cluster 1
ax1 = plt.subplot(gs[-1, 0])
ax1.imshow(kupka_paintings[0])
ax1.title.set_text('Cluster 1') 
ax1.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)

for position in ['bottom', 'top', 'right', 'left']:
    ax1.spines[position].set_color(palette[0])
    ax1.spines[position].set_linewidth(edge_width)


# ax2: cluster2
ax2 = plt.subplot(gs[-1, -4])
ax2.imshow(kupka_paintings[1])
ax2.title.set_text('Cluster 2') 
ax2.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)

for position in ['bottom', 'top', 'right', 'left']:
    ax2.spines[position].set_color(palette[1])
    ax2.spines[position].set_linewidth(edge_width)

# ax3: cluster3
ax3 = plt.subplot(gs[-1, -3])
ax3.imshow(kupka_paintings[2])
ax3.title.set_text('Cluster 3') 
ax3.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)

for position in ['bottom', 'top', 'right', 'left']:
    ax3.spines[position].set_color(palette[2])
    ax3.spines[position].set_linewidth(edge_width)


# ax4: cluster4
ax4 = plt.subplot(gs[-1, -2])
ax4.imshow(kupka_paintings[3])
ax4.title.set_text('Cluster 4') 
ax4.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)

for position in ['bottom', 'top', 'right', 'left']:
    ax4.spines[position].set_color(palette[3])
    ax4.spines[position].set_linewidth(edge_width)


# ax5: cluster5
ax5 = plt.subplot(gs[-1, -1])
ax5.imshow(kupka_paintings[4])
ax5.title.set_text('Cluster 5') 
ax5.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)

for position in ['bottom', 'top', 'right', 'left']:
    ax5.spines[position].set_color(palette[4])
    ax5.spines[position].set_linewidth(edge_width)

plt.tight_layout()
plt.savefig('../plots/kupka_k5.png')

# %%
# show prototypes for each k
def show_prototype_images(artist_km):

    img_paths = ['../data/analysis_subset/img/' + img_id + '.jpg' for img_id in artist_km['prot_id'].tolist()]

    paintings = []
    for path in img_paths:
        paintings.append(mpimg.imread(path))

    fig, axs = plt.subplots(5, 5)

    for i, n_clusters, cluster_id in zip(range(len(paintings)), artist_km['k'].tolist(), artist_km['cluster'].tolist()):
        axs[n_clusters-1][cluster_id].imshow(paintings[i], aspect='auto')

# %%
show_prototype_images(kupka_km)

# %%
# show all paintings belonging to a cluster
kupka['label_k5'] = kupka_plt[-1]['labels']
cl0_ids = kupka.query('label_k5 == 4')['ID'].tolist()

cl_img_paths = ['../data/analysis_subset/img/' + img_id + '.jpg' for img_id in cl0_ids]

cl_paintings = []
for path in cl_img_paths:
    cl_paintings.append(mpimg.imread(path))

fig = plt.figure(figsize=(8, 8))
columns = 5
rows = 2
for i, img in zip(range(1, columns*rows +1), cl_paintings):
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)

plt.show()


