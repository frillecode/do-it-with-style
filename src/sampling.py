'''
Get a sample of paintings to run the style transfer on.
'''
# %%
import os
import re
import shutil
import pandas as pd
import seaborn as sns

# %%
paintings = pd.read_csv('../data/web-gallery-of-art/artwork_dataset.csv')
artists = pd.read_csv('../data/web-gallery-of-art/info_dataset.csv').query('school == "painter"')
path_imgdif = '../data/web-gallery-of-art/artwork'

# %%
artists.groupby('period').size()

# %%
# from realism to impressionism
periods_of_interest = ['Realism', 'Impressionism', 'Romanticism']
a19 = artists.query('period == @periods_of_interest')
p19 = pd.merge(paintings, a19, on='artist', how='inner')


# %%
# split painting metadata
p19[['dating', 'material', 'dimensions']] = p19['picture data'].str.split(',', expand=True).iloc[:, 0:3]
# paintings per material
p19.groupby('material').size().sort_values(ascending=False).head()

# get only oil paintings
materials_of_interest = [' oil on canvas']
p19_oil = p19.query('material == @materials_of_interest')

# %%
p19_oil.groupby('period').size()

# %%
# dating
# p19_oil['dating_clean'] = p19_oil['dating'].astype(int, errors='ignore')
p19_oil['dating_clean'] = p19_oil['dating'].str.extract(r'(\d{4})', expand=False)
p19_oil_dated = p19_oil[~p19_oil['dating_clean'].isna()]
p19_oil_dated['dating_clean'] = p19_oil_dated['dating_clean'].astype(int)

# plot
sns.displot(p19_oil_dated['dating_clean'])

# %%
p19_oil_dated.groupby('period').size()

# %%
# delete unique artists
non_unique_artists = (p19_oil_dated
    .groupby('artist')
    .size()
    .to_frame(name='n')
    .reset_index()
    .query('n > 1')
    ['artist']
    .tolist()
)

p19_oil_dated_nouniq = p19_oil_dated.query('artist == @non_unique_artists')

# %%
# monet subset
monet = p19_oil_dated_nouniq.query('artist == "MONET, Claude"')
monet_ids = monet['ID'].tolist()
monet_paths = [os.path.join(path_imgdif, str(path) + '.jpg') for path in monet_ids]

# # copy paste files
# path_monet_target = '../data/web-gallery-of-art/monet_subset'
# for path in monet_paths:
#     shutil.copy2(src=path, dst=path_monet_target)

# %%
# goya subset
goya = p19_oil_dated_nouniq.query('artist == "GOYA Y LUCIENTES, Francisco de"')
goya_ids = goya['ID'].tolist()
goya_paths = [os.path.join(path_imgdif, str(path) + '.jpg') for path in goya_ids]

# # copy paste files
# path_goya_target = '../data/web-gallery-of-art/goya_subset'
# for path in goya_paths:
#     shutil.copy2(src=path, dst=path_goya_target)


# %%
# van gogh subset
vangogh = p19.query('artist == "GOGH, Vincent van"').query('dimensions == " oil on canvas"')
vangogh['dating_clean'] = vangogh['dating'].str.extract(r'(\d{4})', expand=False)

vangogh_ids = vangogh['ID'].tolist()
vangogh_paths = [os.path.join(path_imgdif, str(path) + '.jpg') for path in vangogh_ids]

# # copy paste files
# path_vangogh_target = '../data/web-gallery-of-art/vangogh_subset'
# for path in vangogh_paths:
#     shutil.copy2(src=path, dst=path_vangogh_target)

# %%
# kupka
kupka_paths = os.listdir('../data/kupka')

kupka_dat = []
for path in kupka_paths:
    fname = re.sub(r'\.jpg', '', path)

    year = re.match(r'\d{4}(?=_)', path).group(0)
    year = int(year)

    name = re.search(r'(?<=_).*(?=\.jpg)', path).group(0)

    kupka_dat.append({
        'ID': fname,
        'artist': 'KUPKA, Frantisek',
        'title': name,
        'dating_clean': year
    })

kupka = pd.DataFrame(kupka_dat)


# %%
# generate metadata
meta_data = pd.concat([monet, goya, vangogh, kupka])

# hand removal of problematic paintings
problematic_paintings_ids = [18422, 18425]
meta_data = meta_data.query('ID != @problematic_paintings_ids')

# export
meta_data.to_csv('../data/analysis_subset/subset_metadata.csv', index=False)
# %%
