import torch

from torch_device_settings import TORCH_DEVICE
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, SubsetRandomSampler

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import data_conversion
import generateSimData

import warnings

def get_xyz_from_dataloader(dataloader):
    for i, data in enumerate(dataloader, 0):
        if i == 0:
            x, y, z, _ = data
        else:
            x_t, y_t, z_t, _ = data
            x = torch.cat((x, x_t), dim=0)
            y = torch.cat((y, y_t), dim=0)
            z = torch.cat((z, z_t), dim=0)
    return x, y, z

def get_simulated_datasets():

    datasets = dict()
    datasets['S_S'] = {'dataset_name':'SJL','continuous_y':False,'is_simulated':True}
    datasets['S_S_multiZ'] = {'dataset_name':'SJLmultiZ','continuous_y':False,'is_simulated':True}
    datasets['S_ZII'] = {'dataset_name':'ZafarWWW2017CaseII','continuous_y':False,'is_simulated':True}
    datasets['S_Z'] = {'dataset_name':'ZafarAISTATS2017','continuous_y':False,'is_simulated':True}
    datasets['S_JL'] = {'dataset_name':'JLcontinuousY','continuous_y':True,'is_simulated':True}
    datasets['S_GM'] = {'dataset_name':'GMcontinuousY','continuous_y':True,'is_simulated':True}

    return datasets

def get_datasets_from_file():

    datasets = dict()
    datasets['F_P0'] = {'dataset_name': 'propublica', 'data_filename':'data/compas-scores-two-years-JohndrowLum.csv','continuous_y':False,'is_simulated': False, 'protected': None}
    datasets['F_P1'] = {'dataset_name': 'propublica', 'data_filename':'data/compas-scores-two-years-JohndrowLumMore.csv','continuous_y':False,'is_simulated':False, 'protected': None}
    datasets['F_A'] = {'dataset_name': 'adult_uci', 'data_filename':'data/adult_uci.data','continuous_y':False,'is_simulated':False, 'protected': None}
    datasets['F_A_gender'] = {'dataset_name': 'adult_uci', 'data_filename':'data/adult_uci.data','continuous_y':False,'is_simulated':False, 'protected': 'sex'}
    datasets['F_A_race'] = {'dataset_name': 'adult_uci', 'data_filename':'data/adult_uci.data','continuous_y':False,'is_simulated':False, 'protected': 'race'}

    return datasets

def print_simulated_datasets():

    ds = get_simulated_datasets()
    print('Available simulated datasets:')

    for d in ds:
        print('id={}: full_name={}; continuous_y={}'.format(d,ds[d]['dataset_name'],ds[d]['continuous_y']))
    print('\n')


def print_datasets_from_file():

    df = get_datasets_from_file()
    print('Available datasets from file:')

    for d in df:
        print('id={}: full_name={}; continuous_y={}; filename={}'.format(d,df[d]['dataset_name'],df[d]['continuous_y'],df[d]['data_filename']))
    print('\n')


def get_dataset_by_id(id, debias_type='regression_cond_y', number_of_continuous_y_intervals=None, desired_number_of_samples=5000, visualize_data=False):
    """

    :param id: id of the dataset (string); current options: 'S_S', 'S_ZII', 'S_Z', 'S_JL', 'S_GM', 'F_P0', 'F_P1', 'F_P2', 'F_S', 'F_A' (prefix S/F: simulated/file)
    :param debias_type: current options: 'lda', 'regression_cond_x', 'regression_cond_y'
    :param number_of_continuous_y_intervals: continuous y intervals to use (only supported by regression_cond_y)
    :param desired_number_of_samples:  desired number of samples for synthetic dataset
    :param visualize_data: if supported, the dataset is visualized when it is loaded
    :return: current_dataset, x, y, z, xl, yl, zl
    """

    datasets_sim = get_simulated_datasets()
    datasets_file = get_datasets_from_file()

    if (id in datasets_sim) and (id in datasets_file):
        raise ValueError('Ambiguous dataset id: {}; could be simulated or from file'.format(id))
    elif (id in datasets_sim):
        simulated = True
    elif (id in datasets_file):
        simulated = False
    else:
        print_simulated_datasets()
        print_datasets_from_file()
        raise ValueError('Could not find id: {}'.format(id))

    if simulated:
        dataset = datasets_sim[id]
        if (number_of_continuous_y_intervals is not None) and (not dataset['continuous_y']):
            print('INFO: dataset is not continuous, so number of continuous values will be ignored')

        use_continuous_y = dataset['continuous_y']
        current_dataset, x, y, z, xl, yl, zl = get_dataset(simulator_dataset_name=dataset['dataset_name'],
                                                           nr_of_samples=desired_number_of_samples,
                                                           debias_type=debias_type,
                                                           use_continuous_y=use_continuous_y,
                                                           number_of_continuous_y_intervals=number_of_continuous_y_intervals,
                                                           visualize_data=visualize_data)
    else:
        dataset = datasets_file[id]
        if (number_of_continuous_y_intervals is not None) and (not dataset['continuous_y']):
            print('INFO: dataset is not continuous, so number of continuous values will be ignored')

        use_continuous_y = dataset['continuous_y']
        current_dataset, x, y, z, xl, yl, zl = get_dataset(dataset_name=dataset['dataset_name'],
                                                           data_filename=dataset['data_filename'],
                                                           protected=dataset['protected'],
                                                           debias_type=debias_type,
                                                           use_continuous_y=use_continuous_y,
                                                           number_of_continuous_y_intervals=number_of_continuous_y_intervals,
                                                           visualize_data=visualize_data)

    return current_dataset, x, y, z, xl, yl, zl, use_continuous_y


def get_data_loader_from_given_data(x,y,z,use_continuous_y=False,desired_batch_size=None):

    # todo: maybe make sure that the input already has the right dimension; temporary hack for now
    if len(x.size())==2:
        x_eff = x.reshape([x.size()[0],1,x.size()[1]])
    else:
        x_eff = x

    if len(y.size()) == 2:
        y_eff = y.reshape([y.size()[0], 1, y.size()[1]])
    else:
        y_eff = y

    if len(z.size()) == 2:
        z_eff = z.reshape([z.size()[0], 1, z.size()[1]])
    else:
        z_eff = z

    current_dataset = GenericDataset(x=x_eff, y=y_eff, z=z_eff, use_continuous_y=use_continuous_y)
    data_loader = DataLoader(current_dataset, batch_size=desired_batch_size, shuffle=True, num_workers=1)

    return data_loader

def get_data_loaders(dataset,x,y,z,desired_batch_size=None,desired_batch_size_training=None,desired_batch_size_testing=None,testing_percent=50):
    # split data for training and testing
    num_data = len(dataset)
    indices = range(num_data)
    x_train, x_test, y_train, y_test, z_train, z_test, training_idx, testing_idx \
        = train_test_split(x.cpu().numpy(), y.cpu().numpy(), z.cpu().numpy(), indices, test_size=testing_percent / 100)

    nr_training = len(training_idx)
    nr_testing = len(testing_idx)

    print('\nData split: # of samples: ({},{}) training/testing\n'.format(nr_training, nr_testing))

    if not (desired_batch_size is None) and ( not(desired_batch_size_testing is None) or not(desired_batch_size_training is None)):
        warnings.warn('Either only desired_batch_size should be set or desired_batch_size_testing AND desired_batch_size_training; using desired_batch_size for training and testing now')
        desired_batch_size_training = desired_batch_size
        desired_batch_size_testing = desired_batch_size

    if not ( desired_batch_size is None ):
        desired_batch_size_training = desired_batch_size
        desired_batch_size_testing = desired_batch_size

    if (desired_batch_size_testing is None):
        desired_batch_size_testing = nr_testing
        print('INFO testing batch size was not set. Setting to single batch with batch size: {}'.format(desired_batch_size_testing))

    if (desired_batch_size_training is None):
        desired_batch_size_training = nr_training
        print('INFO training batch size was not set. Setting to single batch with batch size: {}'.format(desired_batch_size_training))

    # define our samplers -- we use a SubsetRandomSampler because it will return
    # a random subset of the split defined by the given indices without replacement
    train_sampler = SubsetRandomSampler(training_idx)
    testing_sampler = SubsetRandomSampler(testing_idx)

    if desired_batch_size is None:
        # TODO: why?
        desired_batch_size = max(nr_training,nr_testing)

    if (nr_training>desired_batch_size_training):
        warnings.warn('Currently only un-batched training is fully supported. You are requesting a batch size of ' + str(desired_batch_size_training) + '. Expect possible issues.')
        #raise ValueError('Currently only un-batched training is supported')


    # TODO: have encountered the necessity to set num_workers = 0 instead of 1 to avoid a weird debugging issue
    # create a data-loaders
    train_loader = DataLoader(dataset, batch_size=desired_batch_size_training, sampler=train_sampler, shuffle=False, num_workers=0)
    testing_loader = DataLoader(dataset, batch_size=desired_batch_size_testing, sampler=testing_sampler, shuffle=False, num_workers=0)

    return train_loader, testing_loader


def get_continuous_y_intervals(y, number_of_continuous_y_intervals=10):
    y_np = y.cpu().numpy()
    desired_percentiles = np.linspace(0, 100, number_of_continuous_y_intervals + 1)
    val_ranges = []
    for d in desired_percentiles[1:]:
        current_percentile = np.percentile(y_np, d)
        val_ranges.append(float(current_percentile))
    print('INFO: Computed y percentiles: {}'.format(val_ranges))

    return val_ranges

#todo: create base class for these datasets

def compute_group_indices(y,y_val_ranges):
    y_group_indices = dict()

    if y_val_ranges is not None:

        for i, v in enumerate(y_val_ranges):
            if i == 0:
                current_y_idx = (y <= v)
            else:
                current_y_idx = (y <= v) * (y > y_val_ranges[i - 1])

            # now store them
            y_group_indices[v] = current_y_idx

    return y_group_indices

# define the dataset
class GenericDatasetWithYIntervals(Dataset):
    """generic dataset, but also keeps track of group indices based on y value ranges"""

    def __init__(self, x, y, z=None, y_val_ranges=None, use_continuous_y=False, transform=None):
        """

        :param x: input
        :param y: output classes
        :param z: protected
        :param transform: n/a
        """
        self.x = x
        self.y = y
        self.z = z
        self.transform = transform

        self.use_continuous_y = use_continuous_y
        self.y_group_indices = None
        self.y_val_ranges = y_val_ranges

        self.y_group_indices = compute_group_indices(y=y,y_val_ranges=y_val_ranges)

    def continuous_y(self):
        return self.use_continuous_y

    def get_y_val_ranges(self):
        return self.y_val_ranges

    def get_group_names(self):
        if self.y_group_indices is not None:
            return self.y_group_indices.keys()
        else:
            return None

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):

        sample_x = self.x[idx,...]
        sample_y = self.y[idx,...]

        group_ind_sample = None
        if self.y_group_indices is not None:
            group_ind_sample = []
            for k in self.y_group_indices:
                group_ind_sample.append(self.y_group_indices[k][idx])

        if self.z is not None:
            sample_z = self.z[idx,...]
        else:
            sample_z = None

        if self.transform:
            pass

        return sample_x,sample_y,sample_z,group_ind_sample

class GenericDataset(Dataset):
    """generic dataset."""

    def __init__(self, x, y, z=None, use_continuous_y=False, transform=None):
        """

        :param x: input
        :param y: output classes
        :param z: protected
        :param transform: n/a
        """
        self.x = x
        self.y = y
        self.z = z
        self.transform = transform

        self.use_continuous_y = use_continuous_y
        self.y_val_ranges = [0.0, 1.0]

    def continuous_y(self):
        return self.use_continuous_y

    def get_y_val_ranges(self):
        return self.y_val_ranges

    def get_group_names(self):
        return None

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):

        sample_x = self.x[idx,...]
        sample_y = self.y[idx,...]

        if self.z is not None:
            sample_z = self.z[idx,...]
        else:
            sample_z = None

        group_ind_sample = -1

        if self.transform:
            pass

        return sample_x,sample_y,sample_z,group_ind_sample

class GenericDatasetWithoutProtected(Dataset):
    """generic dataset."""

    def __init__(self, x, y, use_continuous_y=False, transform=None):
        """

        :param x: input
        :param y: output classes
        :param transform: n/a
        """
        self.x = x
        self.y = y
        self.transform = transform

        self.use_continuous_y = use_continuous_y

        self.y_val_ranges = [0.0, 1.0]

    def continuous_y(self):
        return self.use_continuous_y

    def get_y_val_ranges(self):
        return self.y_val_ranges

    def get_group_names(self):
        return None

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):

        sample_x = self.x[idx,...]
        sample_y = self.y[idx,...]

        if self.transform:
            pass

        return sample_x,sample_y

def _insert_channel(x):

    sz = x.shape
    if len(sz)==1:
        y = x.reshape(sz[0],1,1)
    elif len(sz)==2:
        y = x.reshape(sz[0],1,sz[1])
    else:
        raise ValueError('Unsupported dimension')
    return y

def _get_label_range(base_label,nr):
    labels = []
    for n in range(nr):
        current_label = '{}{}'.format(base_label,n)
        labels.append(current_label)
    return labels


def get_dataset(dataset_name=None,data_filename=None, protected=None, simulator_dataset_name=None, standardize_input=True, nr_of_samples=4000, ignore_protected=False,
                debias_type=None, use_continuous_y=None,number_of_continuous_y_intervals=10,y_val_ranges=None,visualize_data=False):

    if dataset_name is not None and simulator_dataset_name is not None:
        raise ValueError('dataset_name and simulator_dataset_name cannot be specified at the same time')

    if dataset_name is not None:

        if dataset_name.lower()=='propublica':
            x, y, z, xl, yl, zl = data_conversion.reformat_propublica_data(data_filename=data_filename)
        elif dataset_name.lower()=='simulation':
            x, y, z, xl, yl, zl = data_conversion.reformat_simulation_data(data_filename=data_filename)
        elif dataset_name.lower()=='adult_uci':
            x, y, z, xl, yl, zl = data_conversion.reformat_adult_uci_data(data_filename=data_filename,protected=protected)
        else:
            raise ValueError('Unknow dataset name: {}'.format(dataset_name))

        # TODO: M's code standardizes the entire dataset all at once. Typically, it proceeds like the following
        # standardize the data
        # scaler = preprocessing.StandardScaler().fit(x_train)
        # scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
        # x_train = x_train.pipe(scale_df, scaler)
        # x_test = x_test.pipe(scale_df, scaler)
        # Thus we are being too generous with the code below...
        if standardize_input:
            std_scale = preprocessing.StandardScaler().fit(x)
            x = std_scale.transform(x)

    elif simulator_dataset_name is not None:
        # NOTE: x,z,y IS the correct ordering!
        x,z,y = generateSimData.generateSimData(n=nr_of_samples,sim=simulator_dataset_name,visualize_data=visualize_data)

        if standardize_input:
            std_scale = preprocessing.StandardScaler().fit(x)
            x = std_scale.transform(x)

        xs = x.shape
        if len(xs)>1:
            nr_x = x.shape[1]
        else:
            nr_x = 1

        ys = y.shape
        if len(ys)>1:
            nr_y = y.shape[1]
        else:
            nr_y = 1

        zs = z.shape
        if len(zs)>1:
            nr_z = z.shape[1]
        else:
            nr_z = 1

        xl = _get_label_range('x',nr_x)
        yl = _get_label_range('y',nr_y)
        zl = _get_label_range('z',nr_z)
    else:
        raise ValueError('No dataset specified')

    # reshape everything so that we conform to the typical pytorch format BxCxX
    x = _insert_channel(x)
    y = _insert_channel(y)
    z = _insert_channel(z)

    # now convert it to torch
    x = torch.from_numpy(x).float().to(TORCH_DEVICE)
    y = torch.from_numpy(y).float().to(TORCH_DEVICE)
    z = torch.from_numpy(z).float().to(TORCH_DEVICE)

    use_y_intervals = False
    if debias_type is not None:
        admissible_debias_types = ['lda', 'regression_cond_x', 'regression_cond_y','causal_inference','pair_subspace']
        if not(debias_type in admissible_debias_types):
            raise ValueError('Unknown debias type {}; must be in {}',format(debias_type,admissible_debias_types))
        if debias_type=='regression_cond_y':
            # need to use GenericDatasetWithYIntervals
            use_y_intervals = True

            if y_val_ranges is None:
                print('INFO: y_val_ranges not specified; computing them from data')
                if use_continuous_y:
                    print('INFO: computing y-intervals')
                    y_val_ranges = get_continuous_y_intervals(y=y, number_of_continuous_y_intervals=number_of_continuous_y_intervals)
                    print('INFO: using the following y-intervals: {}'.format(y_val_ranges))
                else:
                    y_val_ranges = [0.0,1.0]
                    print('INFO: assuming y values are not continous and using default intervals: {}'.format(y_val_ranges))

    if use_y_intervals:

        current_dataset = GenericDatasetWithYIntervals(x=x,y=y,z=z,y_val_ranges=y_val_ranges,use_continuous_y=use_continuous_y)

    else:

        if ignore_protected:
            current_dataset = GenericDatasetWithoutProtected(x=x,y=y,use_continuous_y=use_continuous_y)
        else:
            current_dataset = GenericDataset(x=x, y=y, z=z,use_continuous_y=use_continuous_y)

    return current_dataset, x, y, z, xl, yl, zl
