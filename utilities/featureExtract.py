# External Dependency Imports
import torch
import torch.nn.functional as F

# Internal Project Imports
from utilities.miscellaneous import scl_spatial

def get_feat_norms(x):
    """ Makes l2 norm of x[i,:,j,k] = 1 for all i,j,k. Clamps before sqrt for
    stability
    """
    return torch.clamp(x.pow(2).sum(1, keepdim=True), 1e-8, 1e8).sqrt()


def phi_cat(x, phi, layer_l):
    """ Extract conv features from 'x' at list of VGG16 layers 'layer_l'. Then
        normalize features from each conv block based on # of channels, resize,
        and concatenate into hypercolumns
        Inputs:
            x -- Bx3xHxW pytorch tensor, presumed to contain rgb images
            phi -- lambda function calling a pretrained Vgg16Pretrained model
            layer_l -- layer indexes to form hypercolumns out of
        Outputs:
            features -- BxCxHxW pytorch tensor of hypercolumns extracted from 'x'
                     C depends on 'layer_l'
    """
    h = x.size(2)
    w = x.size(3)

    features = phi(x, layer_l, False)
    # Normalize each layer by # channels so # of channels doesn't dominate 
    # cosine distance
    features = [f / f.size(1) for f in features]

    # Scale layers' features to target size and concatenate
    features = torch.cat([scl_spatial(f, h // 4, w // 4) for f in features], 1) 

    return features

def extract_features(im, phi, augmentation_flip=False):
    """ Extract hypercolumns from 'im' using pretrained VGG16 (passed as phi),
    if speficied, extract hypercolumns from rotations of 'im' as well
        Inputs:
            im -- a Bx3xHxW pytorch tensor, presumed to contain rgb images
            phi -- a lambda function calling a pretrained Vgg16Pretrained model
            augmentation_flip -- whether to extract hypercolumns from rotations of 'im'
                        as well
        Outputs:
            features -- a tensor of hypercolumns extracted from 'im', spatial
                     index is presumed to no longer matter
    """
    # In the original paper used all layers, but dropping conv5 block increases
    # speed without harming quality
    layer_l = [22, 20, 18, 15, 13, 11, 8, 6, 3, 1]
    features = phi_cat(im, phi, layer_l)

    # If specified, extract features from 90, 180, 270 degree rotations of 'im'
    if augmentation_flip:
        augmentation_list = [torch.flip(im, [2]).transpose(2, 3),
                    torch.flip(im, [2, 3]),
                    torch.flip(im, [3]).transpose(2, 3)]

        for i, image_augmentation in enumerate(augmentation_list):
            features_new = phi_cat(image_augmentation, phi, layer_l)

            # Code never looks at patches of features, so fine to just stick
            # features from rotated images in adjacent spatial indexes, since
            # they will only be accessed in isolation
            if i == 1:
                features = torch.cat([features, features_new], 2)
            else:
                features = torch.cat([features, features_new.transpose(2, 3)], 2)

    return features
