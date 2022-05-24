import re
import ndjson
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances



'''
Old method, not as good as kmeans 
'''

# %%
# # chunked kmeans
# def chunk_list(lst, n_chunks):

#     chunk_size = len(lst) // n_chunks
#     chunked_lst = [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

#     return chunked_lst

# def find_painting_by_distance_to_centroid(embeddings, ids, doc_rank=0):

#     km = KMeans(n_clusters=1)
#     km.fit(embeddings)

#     centroid = km.cluster_centers_
#     d_centroid = pairwise_distances(
#         X=centroid,
#         Y=embeddings
#     )

#     # index of document at desired rank
#     d_centroid_argsort = np.argsort(d_centroid)[0]
#     doc_idx = int(np.argwhere(d_centroid_argsort == doc_rank))

#     # get id of prototypical doc
#     prototype_doc_id = ids[doc_idx]
#     inertia = km.inertia_

#     return prototype_doc_id, inertia


# # %%
# # prototypes of chunks
# kupka = kupka.sort_values(by='dating_clean')
# kupka_emb = kupka['embedding'].tolist()
# kupka_ids = kupka['ID'].tolist()

# chunk_prototypes = []
# for k in [1, 2, 3, 4]:
#     chunks_emb = chunk_list(kupka_emb, n_chunks=k)
#     chunks_ids = chunk_list(kupka_ids, n_chunks=k)

#     for i, emb, ids in zip(range(len(chunks_emb)), chunks_emb, chunks_ids):
#         prot_id, inertia = find_painting_by_distance_to_centroid(emb, ids)
#         chunk_prototypes.append({
#             'k': k,
#             'chunk': i,
#             'prot_id': prot_id,
#             'inertia': inertia
#         })

# df_chunk_prototypes = pd.DataFrame(chunk_prototypes)


''' 
Unwrapped prototyping
'''

# %%
# big kmeans
kupka = kupka.sort_values(by='dating_clean')
kupka_emb = kupka['embedding'].tolist()
kupka_ids = kupka['ID'].tolist()

kmeans_prototypes = []
kmeans_coordinates = []
for k in [1, 2, 3, 4, 5]:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(kupka_emb)
    inertia = km.inertia_
    centroids = km.cluster_centers_

    # get centroids
    for cluster_id, centroid in enumerate(centroids):
        d_centroid = pairwise_distances(
            X=centroid.reshape(1, -1),
            Y=kupka_emb
        )

        # index of closest document to centroid
        d_centroid = d_centroid[0]
        doc_idx = np.argmin(d_centroid)

        prototype_doc_id = kupka_ids[doc_idx]
        prototype_doc_coordinates = kupka_emb[doc_idx]

        kmeans_prototypes.append({
            'k': k,
            'inertia': inertia,
            'cluster': cluster_id,
            'prot_id': prototype_doc_id,
            'prot_coord': prototype_doc_coordinates,
            'centroid_coord': centroid.tolist(),
            'distance_to_centroid': d_centroid[doc_idx],
            'median_distance_to_centroid': np.median(d_centroid),
            'n_points_in_cluster': len(np.where(km.labels_ == cluster_id)[0])
        })

df_kmeans_prototypes = pd.DataFrame(kmeans_prototypes)