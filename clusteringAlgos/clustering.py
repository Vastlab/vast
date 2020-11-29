import sys
from .finch import FINCH

import torch
import numpy as np
import time
from sklearn.cluster import DBSCAN
from ..tools import pairwisedistances

def KMeans(x, K=450, verbose=True, init=None, seed=9, *args_passed, **kargs):
    """
    https://github.com/src-d/kmcuda
    kmeans_cuda(samples, clusters, tolerance=0.01, init="k-means++",
                yinyang_t=0.1, metric="L2", average_distance=False,
                seed=time(), device=0, verbosity=0)

    samples numpy array of shape [number of samples, number of features] or tuple(raw device pointer (int), device index (int), shape (tuple(number of samples, number of features[, fp16x2 marker]))). In the latter case, negative device index means host pointer. Optionally, the tuple can be 2 items longer with preallocated device pointers for centroids and assignments. dtype must be either float16 or convertible to float32.

    clusters integer, the number of clusters.

    tolerance float, if the relative number of reassignments drops below this value, algorithm stops.

    init string or numpy array, sets the method for centroids initialization, may be "k-means++", "afk-mc2", "random" or numpy array of shape [clusters, number of features]. dtype must be float32.

    yinyang_t float, the relative number of cluster groups, usually 0.1. 0 disables Yinyang refinement.

    metric str, the name of the distance metric to use. The default is Euclidean (L2), it can be changed to "cos" to change the algorithm to Spherical K-means with the angular distance. Please note that samples must be normalized in the latter case.

    average_distance boolean, the value indicating whether to calculate the average distance between cluster elements and the corresponding centroids. Useful for finding the best K. Returned as the third tuple element.

    seed integer, random generator seed for reproducible results.

    device integer, bitwise OR-ed CUDA device indices, e.g. 1 means first device, 2 means second device, 3 means using first and second device. Special value 0 enables all available devices. The default is 0.

    verbosity integer, 0 means complete silence, 1 means mere progress logging, 2 means lots of output.

    return tuple(centroids, assignments, [average_distance]). If samples was a numpy array or a host pointer tuple, the types are numpy arrays, otherwise, raw pointers (integers) allocated on the same device. If samples are float16, the returned centroids are float16 too.

    """

    sys.path.insert(0, '/home/adhamija/kmcuda/src')
    from libKMCUDA import kmeans_cuda

    np.random.seed(1993)
    torch.manual_seed(0)
    if init is None:
        random_indx = torch.randint(0,x.shape[0],(K,1)).to(x.device)
        if verbose:
            print(f"Initializing with indxs {random_indx}")
        random_indx = random_indx.repeat(1, x.shape[1])
        init = x.gather(0, random_indx).clone()
        init = init.numpy()

    centroids, assignments = kmeans_cuda(x.numpy(), K, init=init, verbosity=verbose, seed=seed)
    centroids = torch.tensor(centroids)
    assignments = torch.tensor(assignments.astype(np.int64))
    assignments = assignments.type(torch.LongTensor)
    print(f"assignments {assignments.shape} centroids {centroids.shape}")
    return centroids, assignments


def pykeops_KMeans(x, K=10, Niter=300, verbose=True, random_indx='first_k'):
    # try:
    #     import pykeops
    #     pykeops.clean_pykeops()          # just in case old build files are still present
    #     pykeops.test_numpy_bindings()    # perform the compilation
    #     pykeops.clean_pykeops()          # just in case old build files are still present
    #     pykeops.test_torch_bindings()    # perform the compilation
    # except:
    #     print(f"Something is wrong with your pykeops installation"
    #           f"More Information at https://www.kernel-operations.io/keops/python/installation.html")
    #     exit()
    from pykeops.torch import LazyTensor
    N, D = x.shape  # Number of samples, dimension of the ambient space

    # K-means loop:
    # - x  is the point cloud,
    # - cl is the vector of class labels
    # - c  is the cloud of cluster centroids
    start = time.time()
    if random_indx == 'first_k':
        c = x[:K, :].clone()  # Simplistic random initialization
    else:
        random_indx = torch.randint(0,x.shape[0],(K,1))
        print(f"Initializing with indxs {random_indx}")
        print(f"random_indx {random_indx.shape}")
        print(f"random_indxrandom_indxrandom_indx {len(set(random_indx.squeeze().tolist()))}")
        random_indx = random_indx.repeat(1, x.shape[1])
        print(f"random_indx {random_indx.shape}")
        print(f"{torch.min(random_indx)} {torch.max(random_indx)}")
        print(f"x {x.shape}")
        c = x.gather(0, random_indx).clone()
        print(f"c {c.shape}")

    x_i = LazyTensor(x[:, None, :])  # (Npoints, 1, D)

    for i in range(Niter):
        c_j = LazyTensor(c[None, :, :])  # (1, Nclusters, D)
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (Npoints, Nclusters) symbolic matrix of squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        Ncl = torch.bincount(cl).type(x.dtype)  # Class weights
        for d in range(D):  # Compute the cluster centroids with torch.bincount:
            c[:, d] = torch.bincount(cl, weights=x[:, d]) / Ncl

    end = time.time()

    if verbose:
        print("K-means example with {:,} points in dimension {:,}, K = {:,}:".format(N, D, K))
        print('Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n'.format(
                Niter, end - start, Niter, (end-start) / Niter))

    # Cluster_labels, Cluster_centers
    return cl, c

    """
    from kmeans_pytorch import kmeans
    cluster_ids_x, cluster_centers = kmeans(
        X=features, num_clusters=25, distance='euclidean', device=torch.device('cuda:0')
    )
    """
    # Cluster_labels, Cluster_centers = clustering.KMeans(features[:100000,:], K=450, random_indx=None)
    # Cluster_labels, Cluster_centers = clustering.KMeans(features, K=450, random_indx=None)
    # Cluster_labels, Cluster_centers = clustering.KMeans(features[:100000,:], K=5, random_indx=None)
    # N, D, K = 10000, 2048, 50
    # x = torch.randn(N, D) / 6 + .5
    # Cluster_labels, Cluster_centers = clustering.KMeans(x, K=K, random_indx=None)


def dbscan(x, distance_metric, eps=0.3, min_samples=10, *args_passed, **kargs):
    """
    x_i = LazyTensor(x[:, None, :])  # (Npoints, 1, D)
    x_j = LazyTensor(x[None, :, :])  # (1, Npoints, D)
    distance_matrix = ((x_i - x_j) ** 2).sum(-1)  # (Npoints, Npoints) symbolic matrix of squared distances
    del x_i,x_j
    """
    x = x.cuda()
    distance_matrix = pairwisedistances.__dict__[distance_metric](x, x)
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1, metric='precomputed').fit(distance_matrix.cpu().numpy())
    indx = torch.tensor(db.core_sample_indices_).cuda()
    if indx.shape[0] == 0:
        indx = torch.tensor([0]).cuda()
    # from IPython import embed;embed();
    indx = indx[:,None].repeat(1, x.shape[1])
    centroids = x.gather(0, indx).cpu().clone()
    return centroids, torch.tensor(db.labels_) #,db.core_sample_indices_

def finch(x, ind_of_interest=-1, *args_passed, **kargs):
    c, num_clust, req_c = FINCH(x.numpy(),verbose=False)
    c = torch.tensor(c[:,ind_of_interest])[:,None].type(torch.LongTensor).repeat(1, x.shape[1])
    centroids = x.gather(0, c)#.cpu().clone()
    return centroids, c[:,ind_of_interest].cuda()
