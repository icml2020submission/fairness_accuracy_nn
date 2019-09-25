# This is an example of how to perform debiasing_sweep

import debiasing
import torch
import numpy as np
from adversarial_wrapper import adversarial_sweep

load_data_for_debiasing_sweep_parameters = {
    'desired_seed': 2019,
    'dataset_id': 'F_P1',
    'show_data': False,
    'desired_number_of_samples': None,
    'desired_batch_size': None,
    'testing_percent': 50
}

debiasing_sweep_parameters = debiasing.load_data_for_debiasing_sweep(**load_data_for_debiasing_sweep_parameters)

proj_loss_weights = torch.cat((torch.Tensor([0.0]), torch.log(torch.linspace(np.exp(0.1), np.exp(0.9), 13)), torch.Tensor([1.0])))

more_debiasing_sweep_parameters = {
    'auto_normalize_energies': True,
    'nr_io': 32,
    'nr_of_layers': 4,
    'dropout_probability': 0.2,
    'debias_type': 'causal_inference',
    'debias_individual_layers_index': [3],
    'learning_rate': 0.001,
    'scheduler_factor': 0.9,
    'scheduler_patience': 10,
    'create_animated_gif': False,
    'save_figures': True,
    'save_base_directory': 'F_P1_figs',
    'scalarization': 'chebychev',
    'desired_number_of_epochs': 500,
    'proj_loss_weights': proj_loss_weights
}

debiasing_sweep_parameters.update(more_debiasing_sweep_parameters)

# causal debiasing
proj_loss_weights, recorded_sweep_results_train, recorded_sweep_results_test, phats_list, propensity_train, propensity_test =\
    debiasing.debiasing_sweep(**debiasing_sweep_parameters)

# adversarial debiasing
lambdas = torch.linspace(0, 20, 15)

adv_recorded_sweep_results_train, adv_recorded_sweep_results_test \
    = adversarial_sweep(debiasing_sweep_parameters['train_loader'],
                        debiasing_sweep_parameters['testing_loader'],
                        lambdas,
                        propensity_train,
                        propensity_test,
                        debiasing_sweep_parameters['save_base_directory'],
                        debiasing_sweep_parameters['nr_io'],
                        debiasing_sweep_parameters['nr_of_layers'],
                        debiasing_sweep_parameters['dropout_probability'])

