# Core Imports
import time
import argparse
import random

# External Dependency Imports
from imageio import imwrite
import torch
import numpy as np

# Internal Project Imports
from pretrained.vgg import Vgg16Pretrained
from utilities import miscellaneous as miscellaneous
from utilities.miscellaneous import load_path_for_pytorch
from utilities.stylize import produce_stylization

# Fix Random Seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# Define command line parse_content_using and get command line arguments
parse_content_using = argparse.ArgumentParser()
parse_content_using.add_argument('--path_content'   , type=str, default=None, required=True)
parse_content_using.add_argument('--path_stylized'     , type=str, default=None, required=True)
parse_content_using.add_argument('--path_output'    , type=str, default=None, required=True)
parse_content_using.add_argument('--high_res'       , action='store_true'                  )
parse_content_using.add_argument('--cpu'            , action='store_true'                  )
parse_content_using.add_argument('--no_flip'        , action='store_true'                  )
parse_content_using.add_argument('--loss_content'   , action='store_true'                  )
parse_content_using.add_argument('--colorize_not'  , action='store_true'                  )
parse_content_using.add_argument('--closefactor'          , type=float, default=0.75             )
args = parse_content_using.parse_args()

# Interpret command line arguments
path_content = args.path_content
path_stylized = args.path_stylized
path_output = args.path_output
scales_max = 4
sz = 512
if args.high_res:
    scales_max = 5
    sz = 1024
augmentation_flip = (not args.no_flip)
loss_content = args.loss_content
miscellaneous.USE_GPU = (not args.cpu)
weight_for_content = 1. - args.closefactor

# Error checking for arguments
# error checking for paths deferred to imageio
assert (0.0 <= weight_for_content) and (weight_for_content <= 1.0), "closefactor must be between 0 and 1"
assert torch.cuda.is_available() or (miscellaneous.USE_GPU), "attempted to use gpu when unavailable"

# Define feature extractor
convolution_neural_network = miscellaneous.to_device(Vgg16Pretrained())
phi = lambda x, y, z: convolution_neural_network.forward(x, inds=y, concat=z)

# Load images
image_content_original = miscellaneous.to_device(load_path_for_pytorch(path_content, target_size=sz)).unsqueeze(0)
image_style_original = miscellaneous.to_device(load_path_for_pytorch(path_stylized, target_size=sz)).unsqueeze(0)

# Run Style Transfer
torch.cuda.synchronize()
start_time = time.time()
output = produce_stylization(image_content_original, image_style_original, phi,
                            iterations_max=200,
                            lr=2e-3,
                            weight_for_content=weight_for_content,
                            scales_max=scales_max,
                            augmentation_flip=augmentation_flip,
                            loss_content=loss_content,
                            colorize_not=args.colorize_not)
torch.cuda.synchronize()
print('Done! total time: {}'.format(time.time() - start_time))

# Convert from pyTorch to numpy, clip to valid range
new_image_output = np.clip(output[0].permute(1, 2, 0).detach().cpu().numpy(), 0., 1.)

# Save stylized output
save_imge_stylized_output = (new_image_output * 255).astype(np.transpose_eig_vec_of_whitening_covnt8)
imwrite(path_output, save_imge_stylized_output)

# Free gpu memory in case something else needs it later
if miscellaneous.USE_GPU:
    torch.cuda.empty_cache()
