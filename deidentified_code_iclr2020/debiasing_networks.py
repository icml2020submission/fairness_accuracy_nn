import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import pdist, squareform
import torch.optim as optim




def _cov(m, current_mean=None, rowvar=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    if current_mean is None:
        m -= torch.mean(m, dim=1, keepdim=True)
    else:
        m -= current_mean.reshape(m.size()[0], 1)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()

def gram_schmidt(R):
    """
    Performs Gram-Schmidt ortho-normalization of columns in matrix R: assumed to be a 2D tensor
    :param R: matrix to be orthonormalized
    :return: orthonormalized matrix
    """

    nr_of_columns = R.size()[1]
    O = None

    for i in range(nr_of_columns):
        ca = R[:,i:i+1]
        cq = ca.clone()
        for j in range(i):
            cq = cq - torch.matmul(O[:,j:j+1].t(),ca)*O[:,j:j+1]

        cq = cq/cq.norm()

        if O is None:
            O = cq
        else:
            O = torch.cat((O,cq),dim=1)

    return O

class LDA(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.9, track_running_stats=False, sigma_reg=0.0):
        super(LDA, self).__init__()

        self.sigma_init_factor = 0.01
        self.sigma_reg = sigma_reg
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats

        #self.w = Parameter(torch.Tensor(num_features))

        if self.track_running_stats:
            self.register_buffer('running_mean_0', torch.zeros(num_features))
            self.register_buffer('running_mean_1', torch.zeros(num_features))
            self.register_buffer('running_sigma_0', self.sigma_init_factor*torch.eye(num_features))
            self.register_buffer('running_sigma_1', self.sigma_init_factor*torch.eye(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean_0', None)
            self.register_parameter('running_mean_1', None)
            self.register_parameter('running_sigma_0', None)
            self.register_parameter('running_sigma_1', None)
            self.register_parameter('num_batches_tracked', None)

        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean_0.zero_()
            self.running_mean_1.zero_()
            self.running_sigma_0[:] = self.sigma_init_factor*torch.eye(self.num_features)
            self.running_sigma_1[:] = self.sigma_init_factor*torch.eye(self.num_features)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        #self.w.data.uniform_()
        ## make it unit length
        #self.w /= self.w.norm()

    def forward(self, h, z,
                x=None,
                y=None,
                propensity_scores=None,
                turn_protected_projection_on=True,
                use_only_projection_penalty=False,
                penalize_fit=False,
                current_epoch=None,
                group_indices=None):

        # z is a binary variable

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1

            h0 = torch.masked_select(h,z==0).reshape(-1,self.num_features)
            h1 = torch.masked_select(h,z==1).reshape(-1,self.num_features)


            if len(h0)>0:
                current_mean_0 = h0.mean(dim=0)
                # detach here, to make sure this does not become part of the computational tree
                self.running_mean_0 = self.momentum*self.running_mean_0 + (1-self.momentum)*current_mean_0
                self.running_mean_0 = self.running_mean_0.detach()
                current_sigma_0 = _cov(h0, current_mean=self.running_mean_0)
                # detach here, to make sure this does not become part of the computational tree
                self.running_sigma_0 = self.momentum * self.running_sigma_0 + (1 - self.momentum) * current_sigma_0
                self.running_sigma_0 = self.running_sigma_0.detach()

            if len(h1)>0:
                current_mean_1 = h1.mean(dim=0)
                # detach here, to make sure this does not become part of the computational tree
                self.running_mean_1 = self.momentum*self.running_mean_1 + (1-self.momentum)*current_mean_1
                self.running_mean_1 = self.running_mean_1.detach()
                current_sigma_1 = _cov(h1,current_mean=self.running_mean_1)
                # detach here, to make sure this does not become part of the computational tree
                self.running_sigma_1 = self.momentum * self.running_sigma_1 + (1 - self.momentum) * current_sigma_1
                self.running_sigma_1 = self.running_sigma_1.detach()

            mean_0 = self.running_mean_0
            mean_1 = self.running_mean_1
            sigma_0 = self.running_sigma_0
            sigma_1 = self.running_sigma_1

        else:
            if self.track_running_stats:
                # just use what we computed before
                mean_0 = self.running_mean_0
                mean_1 = self.running_mean_1
                sigma_0 = self.running_sigma_0
                sigma_1 = self.running_sigma_1
            else:
                # compute values on the fly
                h0 = torch.masked_select(h, z == 0).reshape(-1, self.num_features)
                h1 = torch.masked_select(h, z == 1).reshape(-1, self.num_features)

                mean_0 = h0.mean(dim=0)
                sigma_0 = _cov(h0, current_mean=mean_0)

                mean_1 = h1.mean(dim=0)
                sigma_1 = _cov(h1,current_mean=mean_1)

        # now compute the current LDA direction
        sigma = 0.5*(sigma_0 + sigma_1)

        # DEBUG MODE
        # print this every 50 epochs
        if current_epoch is not None:
            if current_epoch%50==0:
                eigs = sigma.detach().eig()
                #eigs_re_min = eigs[0][:,0].min()
                #eigs_re_max = eigs[0][:,0].max()
                #eigs_im_min = eigs[0][:, 1].min()
                #eigs_im_max = eigs[0][:, 1].max()
                #
                #print('INFO: Sigma: min_re={:.3f}; max_re={:.3f}; min_im={:.3f}; max_im={:.3f}'.format(eigs_re_min,eigs_re_max,eigs_im_min,eigs_im_max))
                print('real(eigs) = {}'.format(eigs[0][:, 0]))

        # todo: maybe use a formulation that does not require the explicit computation of the inverse
        w_unnormalized = torch.matmul(torch.inverse((1.0-self.sigma_reg)*sigma+self.sigma_reg*torch.eye(self.num_features)),(mean_1-mean_0))

        w = w_unnormalized/w_unnormalized.norm()

        return w

    def extra_repr(self):
        return '{num_features}, eps={eps}, sigma_init_factor={sigma_init_factor}, momentum={momentum}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(LDA, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

# one of the possibilities for DebiasLayer based on Linear Discriminant Analysis
class ProtectedLDAProjectionLayer(nn.Module):

    def __init__(self,num_features, sigma_reg=0.0):
        super(ProtectedLDAProjectionLayer, self).__init__()
        self.lda = LDA(num_features=num_features,sigma_reg=sigma_reg)

    def forward(self, h, z,
                x=None,
                y=None,
                propensity_scores=None,
                turn_protected_projection_on=True,
                use_only_projection_penalty=False,
                penalize_fit=False,
                current_epoch=None,
                group_indices=None):

        if turn_protected_projection_on or use_only_projection_penalty:
            w = self.lda(h,z,current_epoch=current_epoch, group_indices=group_indices)

            # now do the projection
            projected_onto_h = torch.matmul(h,w)

            projection_loss = (projected_onto_h**2).sum()

            if use_only_projection_penalty:
                # todo: not sure why this needs to be cloned here to avoid issues with in-place operations for auto-diff
                y = h.clone()
                return y, projection_loss
            else:
                #todo: check this
                #ph = h - torch.ger(projected_onto_h, w)
                ph = h - (torch.ger(projected_onto_h.squeeze(), w)).view_as(h)
                return ph, projection_loss
        else:
            return h, 0.0

# one of the possibilities for DebiasLayer based on linear regression of internal represntation h on x
class ProtectedRegressionLayerCondX(nn.Module):

    def __init__(self):
        super(ProtectedRegressionLayerCondX, self).__init__()

    def forward(self, h, z,
                x=None,
                y=None,
                propensity_scores=None,
                turn_protected_projection_on=True,
                use_only_projection_penalty=False,
                penalize_fit=False,
                current_epoch=None,
                group_indices=None):
        # model is h = b z + Cx + \epsilon
        # we compute b (should be a vector, because z is assumed a scalar)
        # and then compute the sum of norms of bz for all samples, i.e., b^T b \sum_i z_i^2
        # h = [b C][z;x] := [b C] s
        # for all measurements at the same time
        # H = [b C] S w/ S = [s_1,s_2,...,s_s]
        # H S^T(S S^T)^{-1} = [b C]

        sz_z = z.size()
        sz_x = x.size()

        if len(sz_z)!=3 or len(sz_x)!=3:
            raise ValueError('Expected a 3D array for x and z')

        nr_of_protected_variables = sz_z[2]

        if turn_protected_projection_on or use_only_projection_penalty:

            # first remove the channel dimension

            # transpose is the opposite in pytorch convention as batch size is first
            ST = (torch.cat((z,x),dim=2))[:,0,:] # also remove the middle dimension
            S = torch.transpose(ST,dim0=0,dim1=1)

            SST = S.matmul(ST)

            S_inv_SST = ST.matmul(torch.inverse(SST))
            H = torch.transpose((h[:,0,:]).clone(),dim0=0,dim1=1)
            R = H.matmul(S_inv_SST)

            # orthonormalize columns if needed
            w = gram_schmidt(R)
            projected_onto_h = torch.matmul(h.clone(), w)
            projection_loss = (projected_onto_h ** 2).sum()/projected_onto_h.shape[0]

            if penalize_fit:
                residual = H - R.matmul(S)
                projection_loss = projection_loss + (residual ** 2).sum()/projected_onto_h.shape[0]

            if use_only_projection_penalty:
                return h, projection_loss
            else:
                h_projected = h - projected_onto_h.matmul(w.t())
                return h_projected, projection_loss

        else:
            return h, 0.0

# one of the possibilities for DebiasLayer based on linear regression of internal represntation h on y
class ProtectedRegressionLayerCondY(nn.Module):

    def __init__(self):
        super(ProtectedRegressionLayerCondY, self).__init__()

    def forward(self, h, z,
                x=None,
                y=None,
                propensity_scores=None,
                turn_protected_projection_on=True,
                use_only_projection_penalty=False,
                penalize_fit=False,
                current_epoch=None,
                group_indices=None):

        if y is None:
            raise ValueError('Model needs y as input')

        sz_z = z.size()

        if len(sz_z) != 3:
            raise ValueError('Expected a 3D array for z')

        z_stacked = torch.cat((z, y), dim=2)


        if turn_protected_projection_on or use_only_projection_penalty:

            # model is h = b z + Cy + \epsilon

            # first remove the channel dimension
            # transpose is the opposite in pytorch convention as batch size is first
            ST = z_stacked[:,0,:] # also remove the middle dimension
            S = torch.transpose(ST,dim0=0,dim1=1)
            SST = S.matmul(ST)
            S_inv_SST = ST.matmul(torch.inverse(SST))
            H = torch.transpose((h[:,0,:]).clone(),dim0=0,dim1=1)
            R = H.matmul(S_inv_SST) # R is [b C]
            Rsubset = R[:, 0:(R.shape[1]-1)] # extract b

            # orthonormalize columns if needed
            w = gram_schmidt(Rsubset)
            projected_onto_h = torch.matmul(h.clone(), w)
            projection_loss = (projected_onto_h ** 2).sum()

            if penalize_fit:
                residual = H - R.matmul(S)
                # print('residualnormsquared:{}'.format((residual ** 2).sum()))
                projection_loss += (residual ** 2).sum()

            if use_only_projection_penalty:
                return h, projection_loss
            else:
                h_projected = h-projected_onto_h.matmul(w.t())
                return h_projected, projection_loss

        else:
            return h, 0.0

# one of the possibilities for DebiasLayer based on Bolukbasi idea of pairs of words that differ only in gender
# TODO: need to add documentation
class ProtectedPairSubspaceLayer(nn.Module):

    def __init__(self):
        super(ProtectedPairSubspaceLayer, self).__init__()

    def forward(self, h, z,
                x=None,
                y=None,
                propensity_scores=None,
                turn_protected_projection_on=True,
                use_only_projection_penalty=False,
                penalize_fit=False,
                current_epoch=None,
                group_indices=None):
        # for collection over i=1,...,n of \sum_{j: Y_j \ni Y_i} (h_i-h_j) K((X_i - X_j)/b)
        # find first k rows of SVD( cov(difference vectors) ), call this B for the bias subspace
        # let h_B be the projection of h onto B
        # then h - h_B is the projection on the orthogonal subspace of B
        # want to force h_B to be close to zero

        if y is None:
            raise ValueError('Model needs y as input')
        sz_z = z.size()
        if len(sz_z) != 3:
            raise ValueError('Expected a 3D array for z')


        if turn_protected_projection_on or use_only_projection_penalty:

            # use nonparametric regression to get hdiff
            pairwise_sq_dists = squareform(pdist(x[:, 0, :], 'sqeuclidean'))
            pairwise_sq_dists = torch.Tensor(pairwise_sq_dists).unsqueeze(dim=1)
            # TODO: should make bandwidth selection adaptive
            bandwidth = 10
            K = torch.exp(-pairwise_sq_dists / bandwidth ** 2)
            hdiff = h.clone()
            for i in range(0,h.shape[0]):
                hdiff[i,0,:] = torch.mm(K[i,:],(h[:,0,:] - h[i,0,:]))

            # top PC(s) of collection of h_i^diff, call it beta
            u, s, v = torch.svd(hdiff[:, 0, :].clone())
            beta = v[1, :]
            beta.reshape([beta.shape[0], 1])

            H = torch.transpose((h[:, 0, :]).clone(), dim0=0, dim1=1)

            betabetaT = torch.ger(beta, beta)
            projection_loss = ((betabetaT.matmul(H)) ** 2).sum()

            if use_only_projection_penalty:
                return h, projection_loss
            else:
                h_projected = H - betabetaT.matmul(H)
                return h_projected, projection_loss

        else:
            return h, 0.0

# one of the possibilities for DebiasLayer based on propensity scores
class ProtectedCausalInferenceLayer(nn.Module):

    def __init__(self):
        super(ProtectedCausalInferenceLayer, self).__init__()

    def forward(self, h, z=None,
                x=None,
                y=None,
                propensity_scores=None,
                turn_protected_projection_on=True,
                use_only_projection_penalty=False,
                penalize_fit=None,
                current_epoch=None,
                group_indices=None):



        if turn_protected_projection_on or use_only_projection_penalty:

            sz_z = z.size()

            if len(sz_z) != 3:
                raise ValueError('Expected a 3D array for z')

            # sample estimator of WATE using overlap weights, equation (6) from Li, Lock Morgan, et. al
            w1 = propensity_scores[:, :, 0]  # P(Z=0|X)
            w0 = propensity_scores[:, :, 1]  # P(Z=1|X)
            w1z = w1 * z[:, 0, :]
            w1z_sum = torch.sum(w1z)
            w0znot = w0*(1-z[:,0,:])
            w0znot_sum = torch.sum(w0znot)

            weights = w1z/w1z_sum - w0znot/w0znot_sum
            # weighted_h = weights * h.squeeze(dim=1).clone()
            weighted_h = weights * h.squeeze(dim=1)
            projection_loss = torch.norm(torch.sum(weighted_h, 0))
            if use_only_projection_penalty:
                return h, projection_loss

        else:
            return h, 0.0

# define a generic layer
class DebiasLayer(nn.Module):

    def __init__(self, nr_inputs, nr_outputs, debias_type,
                 train_loader=None,
                 active_unit='relu',
                 use_batch_normalization=True,
                 dropout_probability=0.2,
                 penalize_fit=False,
                 use_protected_projection=False,
                 use_only_projection_penalty=False,
                 sigma_reg=0.0,
                 setz = None):

        super(DebiasLayer, self).__init__()

        self.train_loader = train_loader

        self.use_batch_normalization = use_batch_normalization
        self.use_protected_projection = use_protected_projection
        self.use_only_projection_penalty = use_only_projection_penalty

        # TODO: this shouldn't be hard set
        self.use_dropout = True
        self.dropout_probability = dropout_probability

        if active_unit == 'relu':
            self.active_unit = nn.ReLU(inplace=True)
        elif active_unit == 'elu':
            self.active_unit = nn.ELU(inplace=True)
        elif active_unit == 'leaky_relu':
            self.active_unit = nn.LeakyReLU(inplace=True)
        else:
            self.active_unit = None

        if self.use_dropout:
            self.dropout = nn.Dropout(self.dropout_probability)
        else:
            self.dropout = None

        self.penalize_fit = penalize_fit

        if self.use_batch_normalization:
            #self.normalization = nn.BatchNorm1d(nr_outputs, affine=True) #eps=0.0001, momentum=0.75, affine=True)
            # todo: check that this is the correct way of doing it. We follow the BxCxX pytorch data format
            self.normalization = nn.BatchNorm1d(1, affine=True) #eps=0.0001, momentum=0.75, affine=True)
        else:
            self.normalization = None

        if self.use_protected_projection:

            self.debias_type = debias_type
            admissible_debias_types = ['lda', 'regression_cond_x', 'regression_cond_y', 'causal_inference',
                                       'pair_subspace']
            if not (self.debias_type in admissible_debias_types):
                raise ValueError('Admissible debias types are: {}'.format(admissible_debias_types))

            if self.debias_type=='lda':
                self.protected_projection = ProtectedLDAProjectionLayer(nr_outputs,sigma_reg=sigma_reg)
            elif self.debias_type=='regression_cond_x':
                self.protected_projection = ProtectedRegressionLayerCondX()
            elif self.debias_type=='regression_cond_y':
                self.protected_projection = ProtectedRegressionLayerCondY()
            elif self.debias_type == 'causal_inference':
                self.protected_projection = ProtectedCausalInferenceLayer()
            elif self.debias_type == 'pair_subspace':
                self.protected_projection = ProtectedPairSubspaceLayer()
            else:
                raise ValueError('Unknown debias type = {}'.format(self.debias_type))
        else:
            self.protected_projection = None

        # add bias even if there is bias from batch normalization?
        #self.fc = nn.Linear(nr_inputs, nr_outputs, bias = not self.use_batch_normalization)
        self.fc = nn.Linear(nr_inputs, nr_outputs)

    def forward(self, h, z=None, x=None, y=None,
                propensity_scores=None,
                turn_protected_projection_on=True,
                current_epoch=None,
                group_indices=None):

        proj_loss = 0.0

        h = self.fc(h)

        if self.normalization is not None:
            h = self.normalization(h)

        if self.protected_projection is not None and turn_protected_projection_on:
            if self.use_only_projection_penalty:
                # do not do the projection, but compute the loss
                h, proj_loss = self.protected_projection(h, z, x, y, propensity_scores=propensity_scores,
                                                           turn_protected_projection_on=False,
                                                           use_only_projection_penalty=self.use_only_projection_penalty,
                                                           penalize_fit=self.penalize_fit,
                                                           current_epoch=current_epoch, group_indices=group_indices)
            else:

                h, proj_loss = self.protected_projection(h, z, x, y, propensity_scores=propensity_scores,
                                                           turn_protected_projection_on=turn_protected_projection_on,
                                                           use_only_projection_penalty=self.use_only_projection_penalty,
                                                           current_epoch=current_epoch, group_indices=group_indices)

        if self.active_unit is not None:
            h = self.active_unit(h)

        if self.dropout is not None:
            h = self.dropout(h)

        return h, proj_loss

# define a simple network to test
class SimpleNet(nn.Module):

    def __init__(self, nr_inputs):
        super(SimpleNet, self).__init__()

        nr_io = 10

        self.nr_of_layers = 6

        self.fc1 = nn.Linear(nr_inputs,nr_io)
        self.fc2 = nn.Linear(nr_io,nr_io)
        self.fc3 = nn.Linear(nr_io,nr_io)
        self.fc4 = nn.Linear(nr_io,nr_io)
        self.fc5 = nn.Linear(nr_io,nr_io)
        self.fc6 = nn.Linear(nr_io,1)

        self.bn1 = nn.BatchNorm1d(nr_io)
        self.bn2= nn.BatchNorm1d(nr_io)
        self.bn3 = nn.BatchNorm1d(nr_io)
        self.bn4 = nn.BatchNorm1d(nr_io)
        self.bn5 = nn.BatchNorm1d(nr_io)

        # self.bn1 = nn.BatchNorm1d(1)
        # self.bn2 = nn.BatchNorm1d(1)
        # self.bn3 = nn.BatchNorm1d(1)
        # self.bn4 = nn.BatchNorm1d(1)
        # self.bn5 = nn.BatchNorm1d(1)


    def forward(self, h, z=None, x=None):

        # drop_p = 0.2
        #
        # h = F.dropout(F.relu(self.bn1(self.fc1(h))),drop_p)
        # h = F.dropout(F.relu(self.bn2(self.fc2(h))),drop_p)
        # h = F.dropout(F.relu(self.bn3(self.fc3(h))),drop_p)
        # h = F.dropout(F.relu(self.bn4(self.fc4(h))),drop_p)
        # h = F.dropout(F.relu(self.bn4(self.fc5(h))),drop_p)

        h = F.relu(self.bn1(self.fc1(h)))
        h = F.relu(self.bn2(self.fc2(h)))
        h = F.relu(self.bn3(self.fc3(h)))
        h = F.relu(self.bn4(self.fc4(h)))
        h = F.relu(self.bn5(self.fc5(h)))
        h = self.fc6(h)
        return h


#todo: in principle we only need h and z as inputs; x is simply the first h; but allowing three parameters gives more flexibility for future models

# define a network
class Net(nn.Module):

    def __init__(self,
                 train_loader,
                 active_unit='relu',
                 debias_type=None,
                 penalize_fit=False,
                 use_batch_normalization=False,
                 use_protected_projection=False,
                 use_only_projection_penalty=False,
                 sigma_reg=0.0,
                 dropout_probability=0.2,
                 nr_of_layers=4,
                 nr_io=10,
                 debias_individual_layers_index=None):

        super(Net, self).__init__()

        nr_inputs = train_loader.dataset.x.size()[2]
        print('INFO: Using {} input variables'.format(nr_inputs))

        self.nr_io = nr_io
        self.nr_of_layers = nr_of_layers

        # default is to debias all layers including the output layer
        if debias_individual_layers_index is None:
            self.debias_individual_layers = [use_protected_projection]*self.nr_of_layers
            if debias_type == 'pair_subspace': # for this debias type, can't debias last layer
                self.debias_individual_layers[-1] = False
        else:
            self.debias_individual_layers = [False] * nr_of_layers
            for i in debias_individual_layers_index:
                self.debias_individual_layers[i] = True

        self.nr_of_debiasing_layers = int(np.array(self.debias_individual_layers).sum())

        # for now these layers are all the same, but it is possible to specify them separately by directly specifying the lists
        nr_of_output_features_in_layers = [self.nr_io]*self.nr_of_layers
        #use_protected_projection_in_layers = [use_protected_projection]*self.nr_of_layers # this allows if desired to select only some of them for projection

        self.use_batch_normalization = use_batch_normalization
        self.use_protected_projection = use_protected_projection
        self.use_only_projection_penalty = use_only_projection_penalty
        self.sigma_reg = sigma_reg
        self.debias_type = debias_type

        setz = np.unique(train_loader.dataset.z, axis=0)
        setz = torch.from_numpy(setz)

        self.layers = nn.ModuleList()

        for i in range(len(nr_of_output_features_in_layers)-1):
            current_nr_outputs = nr_of_output_features_in_layers[i]
            current_use_protected_projection = self.debias_individual_layers[i]
            if i == 0:
                # first layer
                current_layer = DebiasLayer(nr_inputs=nr_inputs,
                                            nr_outputs=current_nr_outputs,
                                            debias_type=debias_type,
                                            train_loader=train_loader,
                                            active_unit=active_unit,
                                            dropout_probability=dropout_probability,
                                            use_batch_normalization=use_batch_normalization,
                                            use_protected_projection=current_use_protected_projection,
                                            use_only_projection_penalty=use_only_projection_penalty,
                                            sigma_reg=sigma_reg,
                                            setz=setz)
            else:
                # there was a previous layer
                previous_nr_outputs = nr_of_output_features_in_layers[i-1]
                current_layer = DebiasLayer(nr_inputs=previous_nr_outputs,
                                            nr_outputs=current_nr_outputs,
                                            debias_type=debias_type,
                                            train_loader=train_loader,
                                            active_unit=active_unit,
                                            penalize_fit=penalize_fit,
                                            dropout_probability=dropout_probability,
                                            use_batch_normalization=use_batch_normalization,
                                            use_protected_projection=current_use_protected_projection,
                                            use_only_projection_penalty=use_only_projection_penalty,
                                            sigma_reg=sigma_reg,
                                            setz=setz)

            self.layers.append(current_layer)

        last_layer = DebiasLayer(nr_inputs=nr_of_output_features_in_layers[-1],
                                 nr_outputs=1,
                                 debias_type=debias_type,
                                 active_unit=None,
                                 use_batch_normalization=use_batch_normalization,
                                 dropout_probability=dropout_probability,
                                 use_protected_projection=self.debias_individual_layers[i+1],
                                 use_only_projection_penalty=True,
                                 sigma_reg=sigma_reg,
                                 setz=setz)

        self.layers.append(last_layer)

    def forward(self, h, z=None, x=None, y=None,
                turn_protected_projection_on=True,
                current_epoch=None,
                group_indices=None,
                propnet=None,
                propnet_temp=None):

        if x is None:
            # just make it the first input
            x = h.clone()
            x.requires_grad = True

        if propnet is None:
            propensity_scores = None
        else:
            propnet.eval()
            with torch.no_grad():
                propensity_scores = nn.functional.softmax(propnet(x)/propnet_temp, dim=2)

        # now apply all the layers
        projection_loss = 0.0
        for current_layer in self.layers:
            h, current_projection_loss = current_layer(h, z, x, y,
                                                      propensity_scores=propensity_scores,
                                                      turn_protected_projection_on=turn_protected_projection_on,
                                                      current_epoch=current_epoch,
                                                      group_indices=group_indices)
            projection_loss += current_projection_loss

        # TODO: Not sure this is necessary, disabling for the moment
        # if self.nr_of_debiasing_layers == 0:
        #     projection_loss = projection_loss
        # else:
        #     projection_loss = projection_loss/self.nr_of_debiasing_layers

        return h, projection_loss
