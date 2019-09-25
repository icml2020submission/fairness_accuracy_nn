import torch
import numpy as np
import pandas as pd

from torch_device_settings import TORCH_DEVICE

def _code_as_categorical(vals):
    
    dummies = vals.pipe(pd.get_dummies, drop_first=True)

    res_names_raw = dummies.keys()
    res_names = []
    for rn in res_names_raw:
        res_names.append(vals.name + '_' + rn)

    res_values = dummies.values

    return res_values,res_names

def convert_pandas_column_to_numpy(vals,auto_categorical=False,code_as_categorical=False):

    res_names = []

    if auto_categorical:
        unique_vals = vals.unique()
        is_str = type(unique_vals[0]) == type('')

        if not is_str:
            print('Extracting columnn: {}'.format(vals.name))
            res_names.append(vals.name)
            res_values = vals.values.reshape(-1,1)
        else:
            res_values, res_names = _code_as_categorical(vals)

    else:
        if code_as_categorical:
            res_values,res_names = _code_as_categorical(vals)
        else:
            print('Extracting columnn: {}'.format(vals.name))
            res_names.append(vals.name)
            res_values = vals.values.reshape(-1, 1)

    return res_values,res_names

def reformat_simulation_data(data=None,data_filename=None):
    """
    Reformats data, so it gets split into input (x), output (y), and protected (z)
    :param data: assumes a pandas dataframe
    :return:
    """

    if (data is None) and (data_filename is None):
        raise ValueError('Either data or data_filename need to be specified')

    if data is None:
        if data_filename is None:
            raise ValueError('data_filename needs to be specified')
        else:
            data = pd.read_csv(data_filename)

    available_keys = data.keys()

    z_protected_keys = ['z']
    y_output_keys = ['y']

    x_vals = None
    x_labels = []
    y_vals = None
    y_labels = []
    z_vals = None
    z_labels = []

    for k in available_keys:

        if k in z_protected_keys:

            current_values, current_labels = convert_pandas_column_to_numpy(data[k], auto_categorical=True)

            z_labels += current_labels

            if z_vals is None:
                z_vals = current_values
            else:
                z_vals = np.append(z_vals,current_values,axis=1)

        elif k in y_output_keys:
            current_values, current_labels = convert_pandas_column_to_numpy(data[k], auto_categorical=True)

            y_labels += current_labels

            if y_vals is None:
                y_vals = current_values
            else:
                y_vals = np.append(y_vals,current_values,axis=1)

        else: # input variable
            current_values, current_labels = convert_pandas_column_to_numpy(data[k], auto_categorical=True)

            x_labels += current_labels

            if x_vals is None:
                x_vals = current_values
            else:
                x_vals = np.append(x_vals,current_values,axis=1)

    print('\n')

    x_vals = x_vals.astype('float32')
    y_vals = y_vals.astype('float32')
    z_vals = z_vals.astype('float32')

    return x_vals, y_vals, z_vals, x_labels, y_labels, z_labels


def add_value_and_label_data(vals,labels,new_vals,auto_categorical=False):
    current_values, current_labels = convert_pandas_column_to_numpy(new_vals, auto_categorical=auto_categorical)
    if vals is None:
        vals = current_values
    else:
        vals = np.append(vals, current_values, axis=1)
    if labels is None:
        labels = current_labels
    else:
        labels += current_labels
    return vals,labels

def reformat_adult_uci_data(data_filename,protected=None):
    """
    Adapted from: https://github.com/equialgo/fairness-in-ml
    :param data_filename:
    :return:
    """

    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                    'martial_status', 'occupation', 'relationship', 'race', 'sex',
                    'capital_gain', 'capital_loss', 'hours_per_week', 'country', 'target']
    input_data = (pd.read_csv(data_filename, names=column_names,
                              na_values="?", sep=r'\s*,\s*', engine='python').loc[lambda df: df['race'].isin(['White', 'Black'])])

    # sensitive attributes; we identify 'race' and 'sex' as sensitive attributes
    sensitive_attribs = ['race', 'sex']
    z_pd = (input_data.loc[:, sensitive_attribs]
            .assign(race=lambda df: (df['race'] == 'White').astype(int),
                    sex=lambda df: (df['sex'] == 'Male').astype(int)))

    if protected == 'sex':
        z_pd = (z_pd.drop(columns=['race']))
    elif protected == 'race':
        z_pd = (z_pd.drop(columns=['sex']))


    # targets; 1 when someone makes over 50k , otherwise 0
    y_pd = (input_data['target'] == '>50K').astype(int)

    # features; note that the 'target' and sentive attribute columns are dropped
    x_pd = (input_data
         .drop(columns=['target', 'race', 'sex', 'fnlwgt'])
         .fillna('Unknown')
         .pipe(pd.get_dummies, drop_first=True))

    print(f"features x: {x_pd.shape[0]} samples, {x_pd.shape[1]} attributes")
    print(f"targets y: {y_pd.shape} samples")
    print(f"sensitives z: {z_pd.shape[0]} samples, {z_pd.shape[1]} attributes")

    x_vals = None
    x_labels = None
    for x_key in x_pd.keys():
        x_vals,x_labels = add_value_and_label_data(x_vals,x_labels,x_pd[x_key], auto_categorical=True)

    y_vals = y_pd.values
    y_labels = ['target']
    #for y_key in y_pd.keys():
    #    y_vals,y_labels = add_value_and_label_data(y_vals,y_labels,y_pd[y_key], auto_categorical=True)

    z_vals = None
    z_labels = None
    for z_key in z_pd.keys():
        z_vals,z_labels = add_value_and_label_data(z_vals,z_labels,z_pd[z_key], auto_categorical=True)

    x_vals = x_vals.astype('float32')
    y_vals = y_vals.astype('float32')
    z_vals = z_vals.astype('float32')

    return x_vals, y_vals, z_vals, x_labels, y_labels, z_labels

def reformat_propublica_data(data=None,data_filename=None,binarize_protected_variable=True):
    """
    Reformats data, so it gets split into input (x), output (y), and protected (z)
    :param data: assumes a pandas dataframe
    :return:
    """

    if (data is None) and (data_filename is None):
        raise ValueError('Either data or data_filename need to be specified')

    if data is None:
        if data_filename is None:
            raise ValueError('data_filename needs to be specified')
        else:
            data = pd.read_csv(data_filename)

    available_keys = data.keys()

    z_protected_keys = ['race']
    y_output_keys = ['two_year_recid']

    x_vals = None
    x_labels = []
    y_vals = None
    y_labels = []
    z_vals = None
    z_labels = []

    for k in available_keys:

        if k in z_protected_keys:

            if binarize_protected_variable:
                print('INFO: Treating protected variable {} as binary'.format(k))
                z_caucasian = (data['race'] == 'Caucasian').values.astype('float32')
                #z_not_caucasian = (data['race']!='Caucasian').values.astype('float32')
                #current_values = np.append(z_caucasian.reshape(-1,1),z_not_caucasian.reshape(-1,1),axis=1).astype('float32')
                current_values = z_caucasian.reshape(-1,1).astype('float32')

                #current_labels = ['caucasian','not_caucasian']
                current_labels = ['caucasian']
            else:
                current_values, current_labels = convert_pandas_column_to_numpy(data[k], auto_categorical=True)

            z_labels += current_labels

            if z_vals is None:
                z_vals = current_values
            else:
                z_vals = np.append(z_vals,current_values,axis=1)

        elif k in y_output_keys:
            current_values, current_labels = convert_pandas_column_to_numpy(data[k], auto_categorical=True)

            y_labels += current_labels

            if y_vals is None:
                y_vals = current_values
            else:
                y_vals = np.append(y_vals,current_values,axis=1)

        else: # input variable
            current_values, current_labels = convert_pandas_column_to_numpy(data[k], auto_categorical=True)

            x_labels += current_labels

            if x_vals is None:
                x_vals = current_values
            else:
                x_vals = np.append(x_vals,current_values,axis=1)

    print('\n')

    x_vals = x_vals.astype('float32')
    y_vals = y_vals.astype('float32')
    z_vals = z_vals.astype('float32')

    return x_vals, y_vals, z_vals, x_labels, y_labels, z_labels
