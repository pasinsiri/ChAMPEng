import sys 
import getopt 
import pandas as pd
import numpy as np
import re

def read_input(argv): 
    filename = argv[0]
    input_path = ''
    output_path = ''
    try:
        opts, args = getopt.getopt(argv[1:], 'hi:o', ['ifile=', 'ofile='])
    except getopt.GetoptError:
        print(f'{filename} -i <inputfile> -o <outputfile>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(f'{filename} -i <inputfile> -o <outputfile>')
            sys.exit() 
        elif opt in ('-i', '--ifile'):
            input_path = arg
        elif opt in ('-o', '--ofile'):
            output_path = arg 
    if input_path == '':
        raise ValueError('Input path is undefined')
    elif output_path == '':
        raise ValueError('Output path is undefined') 
    return input_path, output_path

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

def dup_flag(ranks, max_order:int):
    """Check if there is any duplication in each candidate's pair selection

    Args:
        ranks (Pandas Series): A series of ranks

    Returns:
        Int: 0 if there is no duplicate, 1 otherwise
    """
    if ranks.nunique() == max_order:
        return 0
    else: return 1

# TODO: main function
def main(argv, primary_key:str = 'รหัสนิสิต', sorting_col:str = 'Timestamp', col_keyword:str = 'pair'):
    N_PAIR = 18
    input_path, output_path = read_input(argv)
    raw_df = pd.read_csv(input_path)

    # * sort data by timestamp and drop duplicates (keep last record for each primary key)
    raw_df = raw_df.sort_values(by = sorting_col) \
                .drop_duplicates(subset = [primary_key], keep = 'last') \
                .set_index(primary_key)

    # * filter columns
    cols = [c for c in raw_df.columns if col_keyword in c.lower()]
    ranks = raw_df[cols]

    # * rename columns
    ranks.columns = [f'rank {i + 1}' for i in range(len(ranks.columns))]

    # * clean values
    rank_cleaned = apply_to_table(clean_rank_score, ranks, 'rank', 'in')

    # * assert no duplication
    rank_cleaned['dup_flag'] = rank_cleaned.apply(dup_flag, max_order = len(ranks.columns), axis = 1)
    if rank_cleaned['dup_flag'].sum() != 0:
        misrank = rank_cleaned[rank_cleaned['dup_flag'] > 0]
        misrank.to_csv('./res/misrank.csv')
        raise ValueError('There are duplications of pair selection, which are exported to ./res/misrank.csv')

    # TODO: create a new table
    col_list = [' '.join(['pair', str(i)]) for i in range(1, N_PAIR + 1)]

    # * drop duplicate flag from rank_cleaned
    rank_cleaned.drop(['dup_flag'], axis = 1, inplace = True)

    scores = pd.DataFrame(index=rank_cleaned.index, columns=col_list)

    for candidate_id in rank_cleaned.index:
        row = rank_cleaned.loc[candidate_id]
        for rank in row.index:
            # Remove the word "rank"
            rank_number = rank[4:]
            scores.loc[candidate_id, row.loc[rank]] = int(rank_number)
            
    # * reorder columns
    pair_order = [' '.join(['pair', str(i)]) for i in range(1, N_PAIR + 1)]
    scores = scores[pair_order]

    # * save to csv
    scores.to_csv(output_path)
    print(f'Transformation completed, file is saved to {output_path}')
    return

if __name__ == "__main__":
   main(sys.argv, primary_key = 'รหัสนิสิต')