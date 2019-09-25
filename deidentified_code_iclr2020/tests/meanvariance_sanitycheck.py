import numpy as np
import torch
from assessfairness import fairness_measures
from sklearn.manifold import TSNE
import matplotlib
from matplotlib import pyplot as plt


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



n = 100
dim_z = 1
nr_io = 10

# model is h = b z + Cy + \epsilon (always setting U to Y)


# generate z and y

# discrete y
p = 1 / 2
z = np.random.binomial(1, p, size=(n,1))
# continuous x1 and discrete x2
x1 = np.random.normal(0, 1, size=(n,1)) + z - 4  # standard deviation controls how correlated x1 and z are
mu = np.exp(-1 + np.multiply(x1, z) / 2 + x1 / 10 + z / 6)
x2 = np.random.poisson(mu)
x = np.column_stack((x1, x2))
lincomp = x1 + 2 * x2 + z + 2.5
probabilities = np.exp(lincomp) / (1 + np.exp(lincomp))
# binary target
y = np.random.binomial(1, probabilities)




# form h
beta = np.random.normal(size=(dim_z,nr_io))
epsilon = np.random.normal(size=(dim_z,nr_io))
gamma = np.random.normal(size=(dim_z,nr_io))
h = z * beta + y * gamma + epsilon # n by nr_io

# run regression
z_stacked =np.hstack((z,y)) # n by dim_z + dim_y
ST = z_stacked
S = np.transpose(ST)
SST = np.matmul(S,ST)
S_inv_SST = np.matmul(ST, np.linalg.inv(SST))
R = np.matmul(np.transpose(h),S_inv_SST)  # R is [b C]
Rsubset = R[:, 0:(R.shape[1] - 1)]  # extract beta
Rsubset_torch = torch.tensor(Rsubset)

# Gram-Schmidt based projection loss
w = gram_schmidt(Rsubset_torch)
Pbeta = np.matmul(w,np.transpose(w))
tildeh = h - np.matmul(h,Pbeta)
index = np.arange(0,100)
y.resize(n,)
z.resize(n,)

colors = ['red','blue']
embedded_tildeh_y0 = TSNE(n_components=2).fit_transform(tildeh[index[y==0]])
plt.scatter(embedded_tildeh_y0[:,0], embedded_tildeh_y0[:,1],
            c = z[index[y==0]],
            cmap=matplotlib.colors.ListedColormap(colors))
plt.show()
embedded_tildeh_y1 = TSNE(n_components=2).fit_transform(tildeh[index[y==1]])
plt.scatter(embedded_tildeh_y1[:,0], embedded_tildeh_y1[:,1],
            c = z[index[y==1]],
            cmap=matplotlib.colors.ListedColormap(colors))
plt.show()

output = 1 / (1 + np.exp(-tildeh))
z.resize((100,1,1))
mv = fairness_measures(output,y,z)
projected_onto_h = np.matmul(h, w)
projection_loss = (projected_onto_h ** 2).sum() # should be equal to ((tildeh - h) ** 2).sum()


# alternative tildeh
z.resize((n,dim_z))
tildeh = h - np.matmul(z,Rsubset)
output = 1 / (1 + np.exp(-tildeh))
z.resize((100,1,1))
mv = fairness_measures(output,y,z)
projection_loss = ((tildeh - h)  ** 2).sum()


from sklearn.manifold import TSNE
import matplotlib
from matplotlib import pyplot as plt
Pbeta =  w.matmul(torch.transpose(w,dim0=0,dim1=1))
tildeh = h[:,0,:] - h[:,0,:].matmul(Pbeta)
tildeh = tildeh.detach().numpy()
index = np.arange(0,100)
y = y[:,0,0].detach().numpy()
z = z[:,0,0].detach().numpy()
colors = ['red','blue']
embedded_tildeh_y0 = TSNE(n_components=2).fit_transform(tildeh[index[y==0]])
plt.scatter(embedded_tildeh_y0[:,0], embedded_tildeh_y0[:,1],
            c = z[index[y==0]],
            cmap=matplotlib.colors.ListedColormap(colors))
plt.show()
embedded_tildeh_y1 = TSNE(n_components=2).fit_transform(tildeh[index[y==1]])
plt.scatter(embedded_tildeh_y1[:,0], embedded_tildeh_y1[:,1],
            c = z[index[y==1]],
            cmap=matplotlib.colors.ListedColormap(colors))
plt.show()