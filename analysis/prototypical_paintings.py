'''
This script is for plotting the clusters and identifying central embeddings
'''
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
from sklearn.metrics import pairwise_distances, silhouette_score

# %%
# plotting parameters
# 
scale = 1.6

plt.rcParams.update({"text.usetex": False,
                    "font.family": "Times New Roman",
                    "font.serif": "serif",
                    "mathtext.fontset": "cm",
                    "axes.unicode_minus": False,
                    "axes.labelsize": 9*scale,
                    "xtick.labelsize": 9*scale,
                    "ytick.labelsize": 9*scale,
                    "legend.fontsize": 9*scale,
                    'axes.titlesize': 14,
                    "axes.linewidth": 1
                    })
 

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
# helper functions
def find_cluster_prototypes(artist_df, plotting_data=False):
    
    artist_df = artist_df.sort_values(by='dating_clean')
    embeddings = artist_df['embedding'].tolist()
    ids = artist_df['ID'].tolist()

    kmeans_prototypes = []
    _plotting = []
    for k in [1, 2, 3, 4, 5, 6]:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(embeddings)
        inertia = km.inertia_
        adjusted_intertia = inertia / len(embeddings)
        centroids = km.cluster_centers_

        if k > 1:
            silhouette = silhouette_score(embeddings, km.labels_, random_state=42)
        else:
            silhouette = None

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
                'adjusted_intertia': adjusted_intertia,
                'silhouette': silhouette,
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


# show prototypes for each k
def show_prototype_images(artist_km):

    img_paths = ['../data/analysis_subset/img/' + img_id + '.jpg' for img_id in artist_km['prot_id'].tolist()]

    paintings = []
    for path in img_paths:
        paintings.append(mpimg.imread(path))

    fig, axs = plt.subplots(6, 6)

    for i, n_clusters, cluster_id in zip(range(len(paintings)), artist_km['k'].tolist(), artist_km['cluster'].tolist()):
        axs[n_clusters-1][cluster_id].imshow(paintings[i], aspect='auto')


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


# load images
def load_proto_imgs(artist_km, k=None):

    if k:
        artist_km = artist_km.query('k == @k')

    img_paths = ['../data/analysis_subset/img/' + img_id + '.jpg' for img_id in artist_km['prot_id'].tolist()]

    paintings_orig = []
    for path in img_paths:
        paintings_orig.append(mpimg.imread(path))
    
    return paintings_orig


def load_style_imgs(artist_km, k=None):

    if k:
        artist_km = artist_km.query('k == @k')

    img_paths = ['../img/' + img_id + '_at_iteration_1000.png' for img_id in artist_km['prot_id'].tolist()]

    paintings_style = []
    for path in img_paths:
        paintings_style.append(mpimg.imread(path))
    
    return paintings_style


def composite_clustering_figure_w_prototypes(plotting_data, paintings, outpath, k):

    fig = plt.figure(figsize=(10, 10), constrained_layout=True) 
    gs = gridspec.GridSpec(2, k, figure=fig) 

    # parameters
    palette = sns.color_palette('Set2')
    edge_width = 5

    # top part: clustering
    ax0 = plt.subplot(gs[0, :])
    plot_prototype_space(plotting_data, k=k, axis=ax0)

    # ax1: cluster 1
    ax1 = plt.subplot(gs[-1, 0])
    ax1.imshow(paintings[0])
    ax1.title.set_text('Cluster 1') 
    ax1.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)

    for position in ['bottom', 'top', 'right', 'left']:
        ax1.spines[position].set_color(palette[0])
        ax1.spines[position].set_linewidth(edge_width)

    if k > 1:
        # ax2: cluster2
        ax2 = plt.subplot(gs[-1, 1])
        ax2.imshow(paintings[1])
        ax2.title.set_text('Cluster 2') 
        ax2.tick_params(left=False,
                        bottom=False,
                        labelleft=False,
                        labelbottom=False)

        for position in ['bottom', 'top', 'right', 'left']:
            ax2.spines[position].set_color(palette[1])
            ax2.spines[position].set_linewidth(edge_width)

    if k > 2:
        # ax3: cluster3
        ax3 = plt.subplot(gs[-1, 2])
        ax3.imshow(paintings[2])
        ax3.title.set_text('Cluster 3') 
        ax3.tick_params(left=False,
                        bottom=False,
                        labelleft=False,
                        labelbottom=False)

        for position in ['bottom', 'top', 'right', 'left']:
            ax3.spines[position].set_color(palette[2])
            ax3.spines[position].set_linewidth(edge_width)

    if k > 3:
        # ax4: cluster4
        ax4 = plt.subplot(gs[-1, 3])
        ax4.imshow(paintings[3])
        ax4.title.set_text('Cluster 4') 
        ax4.tick_params(left=False,
                        bottom=False,
                        labelleft=False,
                        labelbottom=False)

        for position in ['bottom', 'top', 'right', 'left']:
            ax4.spines[position].set_color(palette[3])
            ax4.spines[position].set_linewidth(edge_width)

    if k > 4:
        # ax5: cluster5
        ax5 = plt.subplot(gs[-1, 4])
        ax5.imshow(paintings[4])
        ax5.title.set_text('Cluster 5') 
        ax5.tick_params(left=False,
                        bottom=False,
                        labelleft=False,
                        labelbottom=False)

        for position in ['bottom', 'top', 'right', 'left']:
            ax5.spines[position].set_color(palette[4])
            ax5.spines[position].set_linewidth(edge_width)
    
    if k > 5:
        # ax5: cluster5
        ax6 = plt.subplot(gs[-1, 5])
        ax6.imshow(paintings[5])
        ax6.title.set_text('Cluster 6') 
        ax6.tick_params(left=False,
                        bottom=False,
                        labelleft=False,
                        labelbottom=False)

        for position in ['bottom', 'top', 'right', 'left']:
            ax6.spines[position].set_color(palette[5])
            ax6.spines[position].set_linewidth(edge_width)

    plt.tight_layout()
    plt.savefig(outpath)


def composite_clustering_figure_w_style_mask_and_prototypes(plotting_data, paintings_orig, paintings_style, outpath, k):

    fig = plt.figure(figsize=(10, 14), constrained_layout=True) 
    gs = gridspec.GridSpec(3, k, figure=fig) 

    # parameters
    palette = sns.color_palette('Set2')
    edge_width = 5

    # top part: clustering
    ax0 = plt.subplot(gs[0, :])
    plot_prototype_space(plotting_data, k=k, axis=ax0)

    # ax1: cluster 1
    ax1 = plt.subplot(gs[-1, 0])
    ax1.imshow(paintings_orig[0])
    ax1.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)

    for position in ['bottom', 'top', 'right', 'left']:
        ax1.spines[position].set_color(palette[0])
        ax1.spines[position].set_linewidth(edge_width)
    
    # ax1s: cluster 1
    ax1s = plt.subplot(gs[1, 0])
    ax1s.imshow(paintings_style[0])
    ax1s.title.set_text('Cluster 1') 
    ax1s.tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)

    for position in ['bottom', 'top', 'right', 'left']:
        ax1s.spines[position].set_color(palette[0])
        ax1s.spines[position].set_linewidth(edge_width)

    if k > 1:
        # ax2: cluster2
        ax2 = plt.subplot(gs[-1, 1])
        ax2.imshow(paintings_orig[1])
        ax2.tick_params(left=False,
                        bottom=False,
                        labelleft=False,
                        labelbottom=False)

        for position in ['bottom', 'top', 'right', 'left']:
            ax2.spines[position].set_color(palette[1])
            ax2.spines[position].set_linewidth(edge_width)
    
        # ax2s: cluster2
        ax2s = plt.subplot(gs[1, 1])
        ax2s.imshow(paintings_style[1])
        ax2s.title.set_text('Cluster 2') 
        ax2s.tick_params(left=False,
                        bottom=False,
                        labelleft=False,
                        labelbottom=False)

        for position in ['bottom', 'top', 'right', 'left']:
            ax2s.spines[position].set_color(palette[1])
            ax2s.spines[position].set_linewidth(edge_width)

    if k > 2:
        # ax3: cluster3
        ax3 = plt.subplot(gs[-1, 2])
        ax3.imshow(paintings_orig[2])
        ax3.tick_params(left=False,
                        bottom=False,
                        labelleft=False,
                        labelbottom=False)

        for position in ['bottom', 'top', 'right', 'left']:
            ax3.spines[position].set_color(palette[2])
            ax3.spines[position].set_linewidth(edge_width)

        # ax3s: cluster3
        ax3s = plt.subplot(gs[1, 2])
        ax3s.imshow(paintings_style[2])
        ax3s.title.set_text('Cluster 3') 
        ax3s.tick_params(left=False,
                        bottom=False,
                        labelleft=False,
                        labelbottom=False)

        for position in ['bottom', 'top', 'right', 'left']:
            ax3s.spines[position].set_color(palette[2])
            ax3s.spines[position].set_linewidth(edge_width)   

    if k > 3:
        # ax4: cluster4
        ax4 = plt.subplot(gs[-1, 3])
        ax4.imshow(paintings_orig[3])
        ax4.tick_params(left=False,
                        bottom=False,
                        labelleft=False,
                        labelbottom=False)

        for position in ['bottom', 'top', 'right', 'left']:
            ax4.spines[position].set_color(palette[3])
            ax4.spines[position].set_linewidth(edge_width)

        # ax4s: cluster4
        ax4s = plt.subplot(gs[1, 3])
        ax4s.imshow(paintings_style[3])
        ax4s.title.set_text('Cluster 4') 
        ax4s.tick_params(left=False,
                        bottom=False,
                        labelleft=False,
                        labelbottom=False)

        for position in ['bottom', 'top', 'right', 'left']:
            ax4s.spines[position].set_color(palette[3])
            ax4s.spines[position].set_linewidth(edge_width)

    if k > 4:
        # ax5: cluster5
        ax5 = plt.subplot(gs[-1, 4])
        ax5.imshow(paintings_orig[4])
        ax5.tick_params(left=False,
                        bottom=False,
                        labelleft=False,
                        labelbottom=False)

        for position in ['bottom', 'top', 'right', 'left']:
            ax5.spines[position].set_color(palette[4])
            ax5.spines[position].set_linewidth(edge_width)

        # ax5s: cluster5
        ax5s = plt.subplot(gs[1, 4])
        ax5s.imshow(paintings_style[4])
        ax5s.title.set_text('Cluster 5') 
        ax5s.tick_params(left=False,
                        bottom=False,
                        labelleft=False,
                        labelbottom=False)

        for position in ['bottom', 'top', 'right', 'left']:
            ax5s.spines[position].set_color(palette[4])
            ax5s.spines[position].set_linewidth(edge_width)
    
    if k > 5:
        # ax6: cluster6
        ax6 = plt.subplot(gs[-1, 5])
        ax6.imshow(paintings_orig[5])
        ax6.tick_params(left=False,
                        bottom=False,
                        labelleft=False,
                        labelbottom=False)

        for position in ['bottom', 'top', 'right', 'left']:
            ax6.spines[position].set_color(palette[5])
            ax6.spines[position].set_linewidth(edge_width)

        # ax6s: cluster6
        ax6s = plt.subplot(gs[1, 5])
        ax6s.imshow(paintings_style[5])
        ax6s.title.set_text('Cluster 6') 
        ax6s.tick_params(left=False,
                        bottom=False,
                        labelleft=False,
                        labelbottom=False)

        for position in ['bottom', 'top', 'right', 'left']:
            ax6s.spines[position].set_color(palette[5])
            ax6s.spines[position].set_linewidth(edge_width)

    plt.tight_layout()
    plt.savefig(outpath)



# %%
# run kupka
kupka_km, kupka_plt = find_cluster_prototypes(kupka, plotting_data=True)
show_prototype_images(kupka_km)
# %%
# kupka orig composite
kupka_k = 3
kupka_paintings = load_proto_imgs(kupka_km, k=kupka_k)
# composite_clustering_figure_w_prototypes(kupka_plt, kupka_paintings, '../plots/clustering/kupka_composite.png', k=kupka_k)
# %%
# kupka style composite
kupka_orig_img = load_proto_imgs(kupka_km, k=kupka_k)
kupka_style_img = load_style_imgs(kupka_km, k=kupka_k)
# composite_clustering_figure_w_style_mask_and_prototypes(kupka_plt, kupka_orig_img, kupka_style_img, '../plots/clustering/kupka_style_composite.png', k=kupka_k)


# %%
# run goya
goya_km, goya_plt = find_cluster_prototypes(goya, plotting_data=True)
show_prototype_images(goya_km)
# %%
# goya orig composite
goya_k = 4
goya_paintings = load_proto_imgs(goya_km, k=goya_k)
# composite_clustering_figure_w_prototypes(goya_plt, goya_paintings, '../plots/clustering/goya_composite.png', k=goya_k)
# %%
# goya style composite
goya_orig_img = load_proto_imgs(goya_km, k=goya_k)
goya_style_img = load_style_imgs(goya_km, k=goya_k)
# composite_clustering_figure_w_style_mask_and_prototypes(goya_plt, goya_orig_img, goya_style_img, '../plots/clustering/goya_style_composite.png', k=goya_k)

# %%
# run moent
monet_km, monet_plt = find_cluster_prototypes(monet, plotting_data=True)
show_prototype_images(monet_km)
# %%
# monet orig composite
monet_k = 4
monet_paintings = load_proto_imgs(monet_km, k=monet_k)
# composite_clustering_figure_w_prototypes(monet_plt, monet_paintings, '../plots/clustering/monet_composite.png', k=monet_k)
# %%
# monet style composite
monet_orig_img = load_proto_imgs(monet_km, k=monet_k)
monet_style_img = load_style_imgs(monet_km, k=monet_k)
# composite_clustering_figure_w_style_mask_and_prototypes(monet_plt, monet_orig_img, monet_style_img, '../plots/clustering/monet_style_composite.png', k=monet_k)

# %%
# van gogh
vangogh_km, vangogh_plt = find_cluster_prototypes(vangogh, plotting_data=True)
show_prototype_images(vangogh_km)
# %%
# vangogh orig composite
vangogh_k = 4
vangogh_paintings = load_proto_imgs(vangogh_km, k=vangogh_k)
# composite_clustering_figure_w_prototypes(vangogh_plt, vangogh_paintings, '../plots/clustering/vangogh_composite.png', k=vangogh_k)
# %%
# vangogh style composite
vangogh_orig_img = load_proto_imgs(vangogh_km, k=vangogh_k)
vangogh_style_img = load_style_imgs(vangogh_km, k=vangogh_k)
# composite_clustering_figure_w_style_mask_and_prototypes(vangogh_plt, vangogh_orig_img, vangogh_style_img, '../plots/clustering/vangogh_style_composite.png', k=vangogh_k)
# %%
# inspection
vangogh['label_k4'] = vangogh_plt[3]['labels'].tolist()
monet['label_k4'] = monet_plt[3]['labels'].tolist()
goya['label_k4'] = goya_plt[3]['labels'].tolist()
kupka['label_k3'] = kupka_plt[2]['labels'].tolist()

# %%
# intertia plot

# scale hack
scale = 2

plt.rcParams.update({"text.usetex": False,
                    "font.family": "Times New Roman",
                    "font.serif": "serif",
                    "mathtext.fontset": "cm",
                    "axes.unicode_minus": False,
                    "axes.labelsize": 9*scale,
                    "xtick.labelsize": 9*scale,
                    "ytick.labelsize": 9*scale,
                    "legend.fontsize": 9*scale,
                    'axes.titlesize': 10*scale,
                    "axes.linewidth": 1
                    })


k_labels = [vangogh_k, monet_k, goya_k, kupka_k]
name_labels = ['van Gogh', 'Monet', 'Goya', 'Kupka']

fig = plt.figure(figsize=(12, 6), constrained_layout=True) 
gs = gridspec.GridSpec(1, 4, figure=fig) 

def generate_axis_inertia(artist_km, i):
    ax = plt.subplot(gs[0, i])
    ax.plot(artist_km['k'], artist_km['silhouette'], c='darkred')
    ax.plot(artist_km['k'], artist_km['adjusted_intertia'], c='darkblue')
    ax.axvline(x=k_labels[i], color='grey', linestyle='--')
    ax.set_ylim([0, 2])
    ax.set_xlim([1, 6])
    ax.set_xticks(range(2, 7, 1))
    ax.set_xlabel('k')
    if i == 0:
        ax.set_ylabel('adjusted interia / silhouette score')
    ax.set_title(f'{name_labels[i]} (k={k_labels[i]})')
    return ax

generate_axis_inertia(vangogh_km, 0)
generate_axis_inertia(monet_km, 1)
generate_axis_inertia(goya_km, 2)
generate_axis_inertia(kupka_km, 3)

plt.savefig('../plots/clustering/cluster_selection.png')



# # %%
# # show all paintings belonging to a cluster
# kupka['label_k5'] = kupka_plt[-1]['labels']
# cl0_ids = kupka.query('label_k5 == 4')['ID'].tolist()

# cl_img_paths = ['../data/analysis_subset/img/' + img_id + '.jpg' for img_id in cl0_ids]

# cl_paintings = []
# for path in cl_img_paths:
#     cl_paintings.append(mpimg.imread(path))

# fig = plt.figure(figsize=(8, 8))
# columns = 5
# rows = 2
# for i, img in zip(range(1, columns*rows +1), cl_paintings):
#     fig.add_subplot(rows, columns, i)
#     plt.imshow(img)

# plt.show()


# %%
