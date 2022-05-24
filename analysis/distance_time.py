# %%
import re
import numpy as np
import pandas as pd
import ndjson
import seaborn as sns

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
# euclidean
def euclidean_dist(x, y):
    d = np.sqrt(
        np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y)
    )
    return d

# %%
# test case: kupka over time
kupka = df.query('artist == "KUPKA, Frantisek"').sort_values(by='dating_clean')

kupka_emb = np.array(kupka['embedding'].tolist())

dist_w1 = []
for i in range(0, len(kupka_emb)):
    d = euclidean_dist(kupka_emb[i], kupka_emb[i+1])
    dist_w1.append(d)

    if i == len(kupka_emb) - 2:
        break

sns.scatterplot(
    x=kupka['dating_clean'].tolist()[:-1],
    y=dist_w1
)
