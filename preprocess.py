import numpy as np
import pandas as pd
import json
import json

INFO_PATH = 'data/Info'
DATA_DIR = 'data'

def get_column_name_mapping(data_df, num_col_idx, cat_col_idx, target_col_idx, column_names = None):
    
    if not column_names:
        column_names = np.array(data_df.columns.tolist())
    

    idx_mapping = {}

    curr_num_idx = 0
    curr_cat_idx = len(num_col_idx)
    curr_target_idx = curr_cat_idx + len(cat_col_idx)

    for idx in range(len(column_names)):

        if idx in num_col_idx:
            idx_mapping[int(idx)] = curr_num_idx
            curr_num_idx += 1
        elif idx in cat_col_idx:
            idx_mapping[int(idx)] = curr_cat_idx
            curr_cat_idx += 1
        else:
            idx_mapping[int(idx)] = curr_target_idx
            curr_target_idx += 1


    inverse_idx_mapping = {}
    for k, v in idx_mapping.items():
        inverse_idx_mapping[int(v)] = k
        
    idx_name_mapping = {}
    
    for i in range(len(column_names)):
        idx_name_mapping[int(i)] = column_names[i]

    return idx_mapping, inverse_idx_mapping, idx_name_mapping


if __name__ == "__main__":
    name = 'synthetic'

    data_dir = f"dataset/{name}"
    data_path = f"{data_dir}/data.csv"
    train_path = f"{data_dir}/train.csv"
    test_path = f"{data_dir}/test.csv"
    info_path = f"{data_dir}/info.json"

    df = pd.read_csv(data_path)
    info = json.load(open(info_path, 'r'))

    columns = df.columns
    column_length = len(columns)

    print("Columns:", columns)


    X = df[columns[:-6]].to_numpy()
    cts_list = info['num_col_idx']
    dst_list = info['cat_col_idx']

    outcome_list = info['outcome_col_idx']
    intervention_list = info['intervention_col_idx']

    new_df = pd.DataFrame()

    cts_columns = []
    dst_columns = []
    for i in range(X.shape[1]):
        if i in cts_list and i not in outcome_list:
            cts_columns.append(columns[i])
            new_df[columns[i]] = X[:,i].astype(np.float32)

    for i in range(X.shape[1]):
        if i not in cts_list and i not in intervention_list:
            dst_columns.append(columns[i])
            new_df[columns[i]] = X[:, i].astype(np.int32)

    # Add intervention column A as categorical
    dst_columns.append('A')
    new_df['A'] = df['A'].to_numpy().astype(np.int32)

    # Add the original outcome columns
    outcome_col_idx = info['outcome_col_idx']
    outcome_col_names = [columns[i] for i in outcome_col_idx]

    # for name in outcome_col_names:
    #     cts_columns.append(name)
    #     new_df[name] = df[name].to_numpy().astype(np.float32)

    # Create consolidated outcome columns based on intervention
    # Get the indices for each intervention value
    int_0_idx = info['int_0_idx']  # Outcomes observed when A=0
    int_1_idx = info['int_1_idx']  # Outcomes observed when A=1

    # Determine the number of outcomes (should be same for both interventions)
    n_outcomes = len(int_0_idx)

    # Create consolidated columns Y0, Y1, ..., Y(N-1)
    for i in range(n_outcomes):
        outcome_name = f'Y{i}'
        col_name_when_0 = f'Y{i}_0'  # Column name when A=0
        col_name_when_1 = f'Y{i}_1'  # Column name when A=1

        # Use the appropriate observed outcome based on intervention value
        new_col = np.where(
            df['A'] == 0,
            df[col_name_when_0],
            df[col_name_when_1]
        ).astype(np.float32)

        new_df[outcome_name] = new_col
        cts_columns.append(outcome_name)

    # Add target column
    dst_columns.append('target')
    new_df['target'] = df['target'].to_numpy().astype(np.int32)

    print("new_df")
    print(new_df.head())
        
    new_df.to_csv(f'{data_dir}/processed.csv', index=False)

    # train/ test split

    n_samples = new_df.shape[0]
    n_train = int(n_samples * 0.8)
    train_df = new_df.iloc[:n_train].copy()
    test_df = new_df.iloc[n_train:].copy()

    # save train and test to csv
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    data_df = pd.read_csv(f"{data_dir}/processed.csv")
    num_data = data_df.shape[0]

    column_names = data_df.columns.tolist()

    print("column_names:", column_names)

    new_indices = list(range(len(column_names)))
    print("new_indices:", new_indices)

    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']

    outcome_idx = new_indices[-1-n_outcomes:-1]
    num_col_idx = num_col_idx + outcome_idx
    target_col_idx = [new_indices[-1]]


    info["num_col_idx"] = num_col_idx
    info["cat_col_idx"] = cat_col_idx
    info["target_col_idx"] = target_col_idx
    info["outcome_col_idx"] = outcome_idx

    # delete two keys, int_0_idx and int_1_idx
    if 'int_0_idx' in info:
        del info['int_0_idx']
    if 'int_1_idx' in info:        
        del info['int_1_idx']

    idx_mapping, inverse_idx_mapping, idx_name_mapping = get_column_name_mapping(data_df, num_col_idx, cat_col_idx, target_col_idx, column_names)

    num_columns = [column_names[i] for i in num_col_idx]
    cat_columns = [column_names[i] for i in cat_col_idx]
    target_columns = [column_names[i] for i in target_col_idx]

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df.columns = range(len(train_df.columns))
    test_df.columns = range(len(test_df.columns))

    col_info = {}

    for col_idx in num_col_idx:
        col_info[col_idx] = {}
        col_info[col_idx]['type'] = 'numerical'
        col_info[col_idx]['max'] = float(train_df[col_idx].max())
        col_info[col_idx]['min'] = float(train_df[col_idx].min())

    for col_idx in cat_col_idx:
        col_info[col_idx] = {}
        col_info[col_idx]['type'] = 'categorical'
        col_info[col_idx]['categorizes'] = list(set(train_df[col_idx]))    

    for col_idx in target_col_idx:
        if info['task_type'] == 'regression':
            col_info[col_idx] = {}
            col_info[col_idx]['type'] = 'numerical'
            col_info[col_idx]['max'] = float(train_df[col_idx].max())
            col_info[col_idx]['min'] = float(train_df[col_idx].min())
        else:
            col_info[col_idx] = {}
            col_info[col_idx]['type'] = 'categorical'
            col_info[col_idx]['categorizes'] = list(set(train_df[col_idx]))      

    info['column_info'] = col_info


    train_df.rename(columns = idx_name_mapping, inplace=True)
    test_df.rename(columns = idx_name_mapping, inplace=True)

    for col in num_columns:
        train_df.loc[train_df[col] == '?', col] = np.nan
    for col in cat_columns:
        train_df.loc[train_df[col] == '?', col] = 'nan'
    for col in num_columns:
        test_df.loc[test_df[col] == '?', col] = np.nan
    for col in cat_columns:
        test_df.loc[test_df[col] == '?', col] = 'nan'


    X_num_train = train_df[num_columns].to_numpy().astype(np.float32)
    X_cat_train = train_df[cat_columns].to_numpy()
    y_train = train_df[target_columns].to_numpy()


    X_num_test = test_df[num_columns].to_numpy().astype(np.float32)
    X_cat_test = test_df[cat_columns].to_numpy()
    y_test = test_df[target_columns].to_numpy()


    save_dir = f'{data_dir}'
    np.save(f'{save_dir}/X_num_train.npy', X_num_train)
    np.save(f'{save_dir}/X_cat_train.npy', X_cat_train)
    np.save(f'{save_dir}/y_train.npy', y_train)

    np.save(f'{save_dir}/X_num_test.npy', X_num_test)
    np.save(f'{save_dir}/X_cat_test.npy', X_cat_test)
    np.save(f'{save_dir}/y_test.npy', y_test)

    train_df[num_columns] = train_df[num_columns].astype(np.float32)
    test_df[num_columns] = test_df[num_columns].astype(np.float32)

    train_df.to_csv(f'{save_dir}/train.csv', index = False)
    test_df.to_csv(f'{save_dir}/test.csv', index = False)

    info['column_names'] = column_names
    info['train_num'] = train_df.shape[0]
    info['test_num'] = test_df.shape[0]

    info['idx_mapping'] = idx_mapping
    info['inverse_idx_mapping'] = inverse_idx_mapping
    info['idx_name_mapping'] = idx_name_mapping 

    metadata = {'columns': {}}
    task_type = info['task_type']
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    for i in num_col_idx:
        metadata['columns'][i] = {}
        metadata['columns'][i]['sdtype'] = 'numerical'
        metadata['columns'][i]['computer_representation'] = 'Float'

    for i in cat_col_idx:
        metadata['columns'][i] = {}
        metadata['columns'][i]['sdtype'] = 'categorical'

    if task_type == 'regression':
        for i in target_col_idx:
            metadata['columns'][i] = {}
            metadata['columns'][i]['sdtype'] = 'numerical'
            metadata['columns'][i]['computer_representation'] = 'Float'
    else:
        for i in target_col_idx:
            metadata['columns'][i] = {}
            metadata['columns'][i]['sdtype'] = 'categorical'

    info['metadata'] = metadata

    with open(f'{save_dir}/info_new.json', 'w') as file:
        json.dump(info, file, indent=4)

    print(f'Processing and Saving {name} Successfully!')

    print(name)
    print('Total', info['train_num'] + info['test_num'])
    print('Train', info['train_num'])
    print('Test', info['test_num'])
    if info['task_type'] == 'regression':
        num = len(info['num_col_idx'] + info['target_col_idx'])
        cat = len(info['cat_col_idx'])
    else:
        cat = len(info['cat_col_idx'] + info['target_col_idx'])
        num = len(info['num_col_idx'])
    print('Num', num)
    print('Cat', cat)