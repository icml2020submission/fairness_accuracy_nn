import debiasing
import itertools
import os
import pickle
import sys
import numpy as np
import torch

def main(taskid):
    """

    :param taskid: has to be an integer between 0 and len(dataloader_experiments)*len(debiasingsweep_experiments)-1
    :return:
    """

    dataloader_config = {
            'dataset_id': ['F_P1'],
            'desired_batch_size': [100, None],
    }
    keys, values = zip(*dataloader_config.items())
    dataloader_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    debiasing_sweep_config = {
        'auto_normalize_energies': [True],
        'nr_io': [2],
        'dropout_probability': [0.2],
        'nr_of_layers': [2],
        'debias_type': ['pair_subspace','regression_cond_y', 'causal_inference'],
        'learning_rate': [0.001],
        'scheduler_factor': [0.9],
        'scheduler_patience': [10],
        'scalarization': ['chebychev'],
        'debias_individual_layers_index': [None]
    }
    keys, values = zip(*debiasing_sweep_config.items())
    debiasingsweep_experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    taskid = int(taskid[0])
    i, j = np.unravel_index(taskid, [len(dataloader_experiments), len(debiasingsweep_experiments)])

    np.random.seed(1011)
    desired_seeds = np.random.randint(0, 2 ** 32, size=(len(dataloader_experiments), len(debiasingsweep_experiments)), dtype=np.uint32)
    desired_seed = desired_seeds[i, j]

    os.system('mkdir hypertuning_dataloader{}_debiasingsweep{}/'.format(i,j))
    with open("hypertuning_dataloader{}_debiasingsweep{}/dataloader_config.txt".format(i,j), "wb") as fp:  # Pickling
        pickle.dump(dataloader_experiments, fp)
    with open("hypertuning_dataloader{}_debiasingsweep{}/debiasing_sweep_config.txt".format(i,j), "wb") as fp:   #Pickling
        pickle.dump(debiasingsweep_experiments, fp)

    dict_load_data = debiasing.load_data_for_debiasing_sweep(
        show_data=False,
        desired_seed=desired_seed,
        desired_number_of_samples=None,
        **dataloader_experiments[i])

    proj_loss_weights = torch.cat(
        (torch.Tensor([0.0]), torch.log(torch.linspace(np.exp(0.2), np.exp(0.9), 10)), torch.Tensor([1.0])))

    dict_load_data.update(debiasingsweep_experiments[j])
    debiasing.debiasing_sweep(desired_seed=desired_seed,
                              proj_loss_weights=proj_loss_weights,
                              save_figures=True,
                              desired_number_of_epochs=500,
                              save_base_directory='hypertuning_dataloader{}_debiasingsweep{}/'.format(i,j),
                              **dict_load_data)



# Run the actual program
if __name__ == "__main__":
  main(sys.argv[1:])



# In terminal type
# python hypertuning.py taskid


# this will unpickle the list of dictionaries
# with open("debiasing_sweep_config.txt", "rb") as fp:   # Unpickling
#     b = pickle.load(fp)