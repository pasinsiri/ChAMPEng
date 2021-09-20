import pandas as pd
import re
import numpy as np

# Parameters
INPUT_FILE = './data/champeng_4_pair_rank.csv'
EXPORT_PATH = './res/pair_selection_v2.csv'
MAX_ORDER = 10
SCORE_LIST = [34,21,13,8,5,3,2,1,1,1]
COLUMN_NAMES = ['timestamp', 'name', 'ID', 'year', 'major']

def clean_rank_score(text):
    """ Get the pair number from each cell of rankings
        If the cell is blank, return nan
    """
    try:
        return 'pair ' + re.search(r"\d+", text).group(0)
    except:
        return np.nan
    
def apply_to_table(func, df, name_condition, mode = 'in'):
    """ Apply a function to every column of a dataset which matches the given condition
        
        If mode = 'in' (default),
        the condition is a column's name must contain name_condition
        
        If mode = 'equal',
        the condition is a column's name must equal to name_condition
        
        If mode = 'not in',
        the condition is a column's name must not contain name_condition
        
        If mode = 'not equal',
        the condition is a column's name must not equal to name_condition
    """
    new = df.copy()
    
    if mode == 'in':
        selected_cols = [col for col in df if name_condition in col]
    elif mode == 'equal':
        selected_cols = [col for col in df if col == name_condition]
    elif mode == 'not in':
        selected_cols = [col for col in df if name_condition not in col]
    elif mode == 'not equal':
        selected_cols = [col for col in df if col != name_condition]
    else:
        return "Error: mode is not defined"
    
    for col in new.columns:
        if col in selected_cols:
            new[col] = df[col].apply(func)
    return new

def dup_flag(ranks):
    """Check if there is any duplication in each candidate's pair selection

    Args:
        ranks (Pandas Series): A series of ranks

    Returns:
        Int: 0 if there is no duplicate, 1 otherwise
    """
    if ranks.nunique() == MAX_ORDER:
        return 0
    else: return 1

assert len(SCORE_LIST) == MAX_ORDER

# ? import data
data = pd.read_csv(INPUT_FILE)

# Rename columns
rank_cols = [''.join(['rank', str(i + 1)]) for i in range(MAX_ORDER)]
COLUMN_NAMES.extend(rank_cols)
data.columns = COLUMN_NAMES

# Remove duplicates (keep last)
data.drop_duplicates(subset = 'ID', keep = 'last', inplace = True)

# Drop unused columns and set index
data.drop(['timestamp', 'major', 'year', 'name'], axis = 1, inplace = True)
data.set_index('ID', inplace = True)
ranks = data[[col for col in data if 'rank' in col]]


rank_cleaned = apply_to_table(clean_rank_score, ranks, 'rank', 'in')

# * Assert no duplication
rank_cleaned['dup_flag'] = rank_cleaned.apply(dup_flag, axis = 1)
assert rank_cleaned['dup_flag'].sum() == 0


# TODO: create a new table
col_list = [' '.join(['pair', str(i)]) for i in range(1, MAX_ORDER + 1)]

# Drop duplicate flag from rank_cleaned
rank_cleaned.drop(['dup_flag'], axis = 1, inplace = True)

scores = pd.DataFrame(index=rank_cleaned.index, columns=col_list)

for candidate_id in rank_cleaned.index:
    row = rank_cleaned.loc[candidate_id]
    for rank in row.index:
        # Remove the word "rank"
        rank_number = rank[4:]
        
        scores.loc[candidate_id, row.loc[rank]] = rank_number
        
# Reorder columns
pair_order = [' '.join(['pair', str(i)]) for i in range(1, MAX_ORDER + 1)]
scores = scores[pair_order]

# Save to csv
scores.to_csv(EXPORT_PATH)