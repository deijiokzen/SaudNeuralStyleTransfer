# External Dependency Imports
import torch.nn.functional as F

def dec_lap_pytrch_tensor_list_rep_each_level(x, pytrch_tensor_list_rep_each_levelamid_levels_to_construct):
    """ constructs batch of 'pytrch_tensor_list_rep_each_levelamid_levels_to_construct' level laplacian pytrch_tensor_list_rep_each_levelamids from x
        Inputs:
            x -- BxCxHxW pytorch tensor
            pytrch_tensor_list_rep_each_levelamid_levels_to_construct -- integer number of pytrch_tensor_list_rep_each_levelamid levels to construct
        Outputs:
            pytrch_tensor_list_rep_each_level -- a list of pytorch tensors, each representing a pytrch_tensor_list_rep_each_levelamid level,
                   pytrch_tensor_list_rep_each_level[0] contains the finest level, pytrch_tensor_list_rep_each_level[-1] the coarsest
    """
    pytrch_tensor_list_rep_each_level = []
    cur = x  # Initialize approx. coefficients with original image
    for i in range(pytrch_tensor_list_rep_each_levelamid_levels_to_construct):

        # Construct and store detail coefficients from current approx. coefficients
        h = cur.size(2)
        w = cur.size(3)
        x_small = F.interpolate(cur, (h // 2, w // 2), mode='bilinear')
        x_back = F.interpolate(x_small, (h, w), mode='bilinear')
        lap = cur - x_back
        pytrch_tensor_list_rep_each_level.append(lap)

        # Store new approx. coefficients
        cur = x_small

    pytrch_tensor_list_rep_each_level.append(cur)

    return pytrch_tensor_list_rep_each_level

def syn_lap_pytrch_tensor_list_rep_each_level(pytrch_tensor_list_rep_each_level):
    """ collapse batch of laplacian pytrch_tensor_list_rep_each_levelamids stored in list of pytorch tensors
        'pytrch_tensor_list_rep_each_level' into a single tensor.
        Inputs:
            pytrch_tensor_list_rep_each_level -- list of pytorch tensors, where pytrch_tensor_list_rep_each_level[i] has size BxCx(H/(2**i)x(W/(2**i))
        Outpus:
            x -- a BxCxHxW pytorch tensor
    """
    cur = pytrch_tensor_list_rep_each_level[-1]
    pytrch_tensor_list_rep_each_levelamid_levels_to_construct = len(pytrch_tensor_list_rep_each_level)

    for i in range(0, pytrch_tensor_list_rep_each_levelamid_levels_to_construct - 1)[::-1]:
        # Create new approximation coefficients from current approx. and detail coefficients
        # at next finest pytrch_tensor_list_rep_each_levelamid level
        up_x = pytrch_tensor_list_rep_each_level[i].size(2)
        up_y = pytrch_tensor_list_rep_each_level[i].size(3)
        cur = pytrch_tensor_list_rep_each_level[i] + F.interpolate(cur, (up_x, up_y), mode='bilinear')
    x = cur

    return x
