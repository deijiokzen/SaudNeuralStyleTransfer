# External Dependency Imports
import torch

def center(x):
    """Subtract the mean of 'x' over leading dimension"""
    return x - torch.mean(x, 0, keepdim=True)

def distances_cos_pairwise(x, y):
    """ Compute all pairwise cosine distances between rows of matrix 'x' and matrix 'y'
        Inputs:
            x -- NxD pytorch tensor
            y -- MxD pytorch tensor
        Outputs:
            d -- NxM pytorch tensor where d[i,j] is the cosine distance between
                 the vector at row i of matrix 'x' and the vector at row j of
                 matrix 'y'
    """
    assert x.size(1) == y.size(1), "can only compute distance between vectors of same length"
    assert (len(x.size()) == 2) and (len(y.size()) == 2), "pairwise distance computation"\
                                                          " assumes input tensors are matrices"

    x_normals = torch.sqrt((x**2).sum(1).view(-1, 1))
    y_normals = torch.sqrt((y**2).sum(1).view(-1, 1))
    y_tensor = torch.transpose(y / y_normals, 0, 1)

    d = 1. - torch.mm(x / x_normals, y_tensor)
    return d

def pairwise_distances_sq_l2(x, y):
    """ Compute all pairwise squared l2 distances between rows of matrix 'x' and matrix 'y'
        Inputs:
            x -- NxD pytorch tensor
            y -- MxD pytorch tensor
        Outputs:
            d -- NxM pytorch tensor where d[i,j] is the squared l2 distance between
                 the vector at row i of matrix 'x' and the vector at row j of
                 matrix 'y'
    """
    assert x.size(1) == y.size(1), "can only compute distance between vectors of same length"
    assert (len(x.size()) == 2) and (len(y.size()) == 2), "pairwise distance computation"\
                                                          " assumes input tensors are matrices"

    x_normals = (x ** 2).sum(1).view(-1, 1)
    y_tensor = torch.transpose(y, 0, 1)
    y_normals = (y ** 2).sum(1).view(1, -1)

    d = -2.0 * torch.mm(x, y_tensor)
    d += x_normals
    d += y_normals

    return d

def distance_l2_pairwise(x, y):
    """ Compute all pairwise l2 distances between rows of 'x' and 'y',
        thresholds minimum of squared l2 distance for stability of sqrt
    """
    d = torch.clamp(pairwise_distances_sq_l2(x, y), min=1e-8)
    return torch.sqrt(d)

def l2_center_pairwise_distance(x, y):
    """ subtracts mean row from 'x' and 'y' before computing pairwise l2 distance between all rows"""
    return distance_l2_pairwise(center(x), center(y))

def cos_center_distance_pairwise(x, y):
    """ subtracts mean row from 'x' and 'y' before computing pairwise cosine distance between all rows"""
    return distances_cos_pairwise(center(x), center(y))

