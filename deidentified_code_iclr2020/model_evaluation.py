import torch
import numpy as np
from assessfairness import fairness_measures
from dataset_factory import get_xyz_from_dataloader

def compute_baseline_accuracy(y):
    baseline_accuracy = float(y.sum()) / float(len(y))
    if baseline_accuracy < 0.5:
        baseline_accuracy = 1.0 - baseline_accuracy
    return baseline_accuracy


def get_correct_incorrect(pred,targ):

    total_number = len(pred)
    total_correct = torch.eq(pred.float(),targ).sum().item()
    total_incorrect = total_number-total_correct
    return total_correct,total_incorrect


def evaluate_net_testing_loader(net,testing_loader,criterion=None):
    """
    As the name would suggest, this function should only be applied to the testing_loader

    :param net:
    :param testing_loader:
    :param criterion:
    :param proj_loss_weight:
    :param c_loss_global_multiplier:
    :param proj_loss_global_multiplier:
    :param turn_protected_projection_on:
    :param scalarization:
    :return: all_probs, all_output_vals, all_ys, all_zs, running_c_loss, fairness_measures
    """
    all_probs = None
    all_output_vals = None
    all_zs = None
    all_ys = None

    if testing_loader.dataset.continuous_y():
        using_continuous_y_intervals = True
    else:
        using_continuous_y_intervals = False

    x, y, z = get_xyz_from_dataloader(testing_loader)
    nr = x.shape[0]

    net.eval()

    # TODO: make this loop properly behaved for mini-batches again, it currently only makes sense if the entire data set is used

    with torch.no_grad():

        running_c_loss = 0.0

        for i, data in enumerate(testing_loader, 0):

            # get the inputs
            x_sample, y_sample, z_sample, group_indices = data

            outputs, _ = net(h=x_sample, z=z_sample, y=y_sample,turn_protected_projection_on=False, group_indices=group_indices)

            if criterion is not None:
                c_loss = criterion(outputs, y_sample.view_as(outputs))
            else:
                c_loss = None

            running_c_loss += c_loss/nr

            if not using_continuous_y_intervals:
                current_probs = torch.sigmoid(outputs)
                if all_probs is None:
                    all_probs = current_probs.detach().cpu().numpy()
                else:
                    all_probs = np.append(all_probs, current_probs.detach().cpu().numpy())
            else:
                current_output_vals = outputs
                if all_output_vals is None:
                    all_output_vals = current_output_vals.detach().cpu().numpy()
                else:
                    all_output_vals = np.append(all_output_vals, current_output_vals.detach().cpu().numpy())

            if all_ys is None:
                all_ys = y_sample.detach().cpu().numpy()
            else:
                all_ys = np.append(all_ys, y_sample.detach().cpu().numpy())

            if all_zs is None:
                all_zs = z_sample.detach().cpu().numpy()
            else:
                all_zs = np.append(all_zs, z_sample.detach().cpu().numpy(),axis=0)

        fairness = fairness_measures(all_probs, all_ys, all_zs, propensity)

    return all_probs,all_output_vals,all_ys,all_zs,running_c_loss, fairness
