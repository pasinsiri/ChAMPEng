import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster import hierarchy
from sklearn.decomposition import PCA

# Parameters
INPUT_FILE = './data/pair_selection.csv'
MAX_ORDER = 10
SCORE_LIST = [13,8,5,3,2,1,1,1,1,1]

assert len(SCORE_LIST) == MAX_ORDER

def replace_nan(data):
    """Replace NaN Value
        If a column contains numerical value, replace it with 0.
        Otherwise, replace it with 'UNKNOWN'"""
    
    df = data.copy()
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(0, inplace = True)
        else:
            df[col].fillna('UNKNOWN', inplace = True)
    return df

df = replace_nan(pd.read_csv(INPUT_FILE))
score_transform = {k: v for k, v in zip(range(1, MAX_ORDER + 1), SCORE_LIST)}
df.replace(score_transform, inplace = True)

# TODO: correlation matrix and dendrogram
pair_score = df[[col for col in df if 'pair' in col]]
corr = pair_score.corr()
corr_table = corr.unstack().reset_index(name = 'Value')
corr_table.columns = ['#1', '#2', 'Value']

den = hierarchy.dendrogram(hierarchy.linkage(corr, method = 'ward'),
                           labels = corr.index,
                           leaf_rotation=90)

# TODO: PCA
pca = PCA(n_components=2).fit(pair_score.T)
pca_score = pca.transform(pair_score.T)

# ? Explained variance ratio
print('{0} % of Variance explained'.format(int(pca.explained_variance_ratio_)))

# ? Plot PCA
# plt.figure(figsize = (12,10))
fig, ax = plt.subplots(figsize = (12, 10))
g = sns.scatterplot(pca_score[:,0], pca_score[:,1], alpha = 0.5)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('Principal Component Analysis with top two axes')

# Annotate label for each dot
for i in range(pca_score.shape[0]):
    plt.annotate(str(i+1), (pca_score[i,0], pca_score[i,1]))

plt.savefig()