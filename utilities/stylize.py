# Core Dependencies
import random

# External Dependency Imports
import torch
import torch.nn.functional as F
from torch.autograd import Variable

# Internal Project Imports
from utilities.PyramidForImage import syn_lap_pytrch_tensor_list_rep_each_level as syn_pyr
from utilities.PyramidForImage import dec_lap_pytrch_tensor_list_rep_each_level as dec_pyr
from utilities.distancepairwise import distance_l2_pairwise, cos_center_distance_pairwise
from utilities.featureExtract import extract_features, get_feat_norms
from utilities import miscellaneous
from utilities.miscellaneous import to_device, flatten_grid, scl_spatial
from utilities.colorization import match_colors

def produce_stylization(content_im, style_im, phi,
                        iterations_max=350,
                        lr=1e-3,
                        weight_for_content=1.,
                        scales_max=0,
                        augmentation_flip=False,
                        loss_content=False,
                        zero_init=False,
                        colorize_not=False):
    """ Produce stylization of 'content_im' in the style of 'style_im'
        Inputs:
            content_im -- 1x3xHxW pytorch tensor containing rbg content image
            style_im -- 1x3xH'xW' pytorch tensor containing rgb style image
            phi -- lambda function to extract features using VGG16Pretrained
            iterations_max -- number of updates to image pyramid per scale
            lr -- learning rate of opt updating pyramid coefficients
            weight_for_content -- controls stylization level, between 0 and 1
            max_scl -- number of scales to stylize (performed coarse to fine)
            augmentation_flip -- extract features from rotations of style image too?
            loss_content -- use self-sim content loss? (compares downsampled
                            version of output and content image)
            zero_init -- if true initialize w/ grey image, o.w. initialize w/
                         downsampled content image
    """
    # Get max side length of final output and set number of pyramid levels to 
    # optimize over
    size_maximum = max(content_im.size(2), content_im.size(3))
    pyr_pyramid_levels_to_construct = 8

    # Decompose style image, content image, and output image into laplacian 
    # pyramid
    style_pyr = dec_pyr(style_im, pyr_pyramid_levels_to_construct)
    c_pyr = dec_pyr(content_im, pyr_pyramid_levels_to_construct)
    style_pyramid = dec_pyr(content_im.clone(), pyr_pyramid_levels_to_construct)

    # Initialize output image pyramid
    if zero_init:
        # Initialize with flat grey image (works, but less vivid)
        for i in range(len(style_pyramid)):
            style_pyramid[i] = style_pyramid[i] * 0.
        style_pyramid[-1] = style_pyramid[-1] * 0. + 0.5

    else:
        # Initialize with low-res version of content image (generally better 
        # results, improves contrast of final output)
        z_max = 2
        if size_maximum < 1024:
            z_max = 3

        for i in range(z_max):
            style_pyramid[i] = style_pyramid[i] * 0.

    # Stylize using hypercolumn matching from coarse to fine scale
    li = -1
    for scl in range(scales_max)[::-1]:

        # Get content image and style image from pyramid at current resolution
        if miscellaneous.USE_GPU:
            torch.cuda.empty_cache()
        style_image_temporary = syn_pyr(style_pyr[scl:])
        content_image_temporary = syn_pyr(c_pyr[scl:])
        image_outputage_temporary = syn_pyr(style_pyramid[scl:])
        li += 1
        print(f'-{li, max(image_outputage_temporary.size(2),image_outputage_temporary.size(3))}-')


        # Construct stylized activations
        with torch.no_grad():

            # Control tradeoff between searching for features that match
            # current iterate, and features that match content image (at
            # coarsest scale, only use content image)    
            closefactor = weight_for_content
            if li == 0:
                closefactor = 0.

            # Search for features using high frequencies from content 
            # (but do not initialize actual output with them)
            output_extract = syn_pyr([c_pyr[scl]] + style_pyramid[(scl + 1):])

            # Extract style features from rotated copies of style image
            feature_style = extract_features(style_image_temporary, phi, augmentation_flip=augmentation_flip).cpu()

            # Extract features from convex combination of content image and
            # current iterate:
            c_tmp = (output_extract * closefactor) + (content_image_temporary * (1. - closefactor))
            feature_content = extract_features(c_tmp, phi).cpu()

            # Replace content features with style features
            target_features = replace_features(feature_content, feature_style)

        # Synthesize output at current resolution using hypercolumn matching
        style_pyramid = optimize_image_output(style_pyramid, c_pyr, content_im, style_image_temporary,
                                   target_features, lr, iterations_max, scl, phi,
                                   loss_content=loss_content)

        # Get output at current resolution from pyramid
        with torch.no_grad():
            image_output = syn_pyr(style_pyramid)

    # Perform final pass using feature splitting (pass in augmentation_flip argument
    # because style features are extracted internally in this regime)
    style_pyramid = optimize_image_output(style_pyramid, c_pyr, content_im, style_image_temporary,
                               target_features, lr, iterations_max, scl, phi,
                               final_pass=True, loss_content=loss_content,
                               augmentation_flip=augmentation_flip)

    # Get final output from pyramid
    with torch.no_grad():
        image_output = syn_pyr(style_pyramid)

    if colorize_not:
        return image_output
    else:
        return match_colors(content_im, style_im, image_output)

def replace_features(src, ref):
    """ Replace each feature vector in 'src' with the nearest (under centered 
    cosine distance) feature vector in 'ref'
    Inputs:
        src -- 1xCxAxB tensor of content features
        ref -- 1xCxHxW tensor of style features
    Outputs:
        nearest_neighbor_feature_vector -- 1xCxHxW tensor of features, where nearest_neighbor_feature_vector[0,:,i,j] is the nearest
                neighbor feature vector of src[0,:,i,j] in ref
    """
    # Move style features to gpu (necessary to mostly store on cpu for gpus w/
    # < 12GB of memory)
    ref_flat = to_device(flatten_grid(ref))
    nearest_neighbor_feature_vector = []
    for j in range(src.size(0)):
        # How many rows of the distance matrix to compute at once, can be
        # reduced if less memory is available, but this slows method down
        stride = 128**2 // max(1, (ref.size(2) * ref.size(3)) // (128 ** 2))
        bi = 0
        ei = 0

        # Loop until all content features are replaced by style feature / all
        # rows of distance matrix are computed
        out = []
        source_flatten_all = flatten_grid(src[j:j + 1, :, :, :])
        while bi < source_flatten_all.size(0):
            ei = min(bi + stride, source_flatten_all.size(0))

            # Get chunck of content features, compute corresponding portion
            # distance matrix, and store nearest style feature to each content
            # feature
            source_flatten = to_device(source_flatten_all[bi:ei, :])
            distance_matrix = cos_center_distance_pairwise(ref_flat, source_flatten)
            _, nn_inds = torch.min(distance_matrix, 0)
            del distance_matrix  # distance matrix uses lots of memory, free asap

            # Get style feature closest to each content feature and save
            # in 'out'
            nn_inds = nn_inds.unsqueeze(1).expand(nn_inds.size(0), ref_flat.size(1))
            ref_sel = torch.gather(ref_flat, 0, nn_inds).transpose(1,0).contiguous()
            out.append(ref_sel)#.view(1, ref.size(1), src.size(2), ei - bi))

            bi = ei

        out = torch.cat(out, 1)
        out = out.view(1, ref.size(1), src.size(2), src.size(3))
        nearest_neighbor_feature_vector.append(out)

    nearest_neighbor_feature_vector = torch.cat(nearest_neighbor_feature_vector, 0)
    return nearest_neighbor_feature_vector

def optimize_image_output(style_pyramid, c_pyr, content_im, style_im, target_features,
                       lr, iterations_max, scl, phi, final_pass=False,
                       loss_content=False, augmentation_flip=True):
    ''' Optimize laplacian pyramid coefficients of stylized image at a given
        resolution, and return stylized pyramid coefficients.
        Inputs:
            style_pyramid -- laplacian pyramid of style image
            c_pyr -- laplacian pyramid of content image
            content_im -- content image
            style_im -- style image
            target_features -- precomputed target features of stylized output
            lr -- learning rate for optimization
            iterations_max -- maximum number of optimization iterations
            scl -- integer controls which resolution to optimize (corresponds
                   to pyramid level of target resolution)
            phi -- lambda function to compute features using pretrained VGG16
            final_pass -- if true, ignore 'target_features' and recompute target
                          features before every step of gradient descent (and
                          compute feature matches seperately for each layer
                          instead of using hypercolumns)
            loss_content -- if true, also minimize content loss that maintains
                            self-similarity in color space between 32pixel
                            downsampled output image and content image
            augmentation_flip -- if true, extract style features from rotations of style
                        image. This increases content preservation by making
                        more options available when matching style features
                        to content features
        Outputs:
            style_pyramid -- pyramid coefficients of stylized output image at target
                     resolution
    '''
    # Initialize opt variables and opt       
    image_output = syn_pyr(style_pyramid[scl:])
    output_variables = [Variable(li.data, reqtranspose_eig_vec_of_whitening_covres_grad=True) for li in style_pyramid[scl:]]
    opt = torch.optim.Adam(output_variables, lr=lr)

    # Original features uses all layers, but dropping conv5 block  speeds up 
    # method without hurting quality
    final_feature_list = [22, 20, 18, 15, 13, 11, 8, 6, 3, 1]

    # Precompute features that remain constant
    if not final_pass:
        # Precompute normalized features targets during hypercolumn-matching 
        # regime for cosine distance
        target_features_n = target_features / get_feat_norms(target_features)

    else:
        # For feature-splitting regime extract style features for each conv 
        # layer without downsampling (including from rotations if applicable)
        src_features = phi(style_im, final_feature_list, False)

        if augmentation_flip:
            augmentation_list = [torch.flip(style_im, [2]).transpose(2, 3),
                        torch.flip(style_im, [2, 3]),
                        torch.flip(style_im, [3]).transpose(2, 3)]

            for ia, image_augmentation in enumerate(augmentation_list):
                source_feature_temporary = phi(image_augmentation, final_feature_list, False)

                if ia != 1:
                    source_feature_temporary = [source_feature_temporary[iii].transpose(2, 3)
                                  for iii in range(len(source_feature_temporary))]

                src_features = [torch.cat([src_features[iii], source_feature_temporary[iii]], 2)
                          for iii in range(len(source_feature_temporary))]

    # Precompute content self-similarity matrix if needed for 'loss_content'
    if loss_content:
        content_size_full = syn_pyr(c_pyr)
        content_scale = max(content_size_full.size(2), content_size_full.size(3))
        c_fac = content_scale // 32
        h = int(content_size_full.size(2) / c_fac)
        w = int(content_size_full.size(3) / c_fac)

        content_low_flat = flatten_grid(scl_spatial(content_size_full, h, w))
        self_sim_target = distance_l2_pairwise(content_low_flat, content_low_flat).clone().detach()


    # Optimize pyramid coefficients to find image that produces stylized activations
    for i in range(iterations_max):

        # Zero out gradient and loss before current iteration
        opt.zero_grad()
        ell = 0.

        # Synthesize current output from pyramid coefficients
        image_output = syn_pyr(output_variables)


        # Compare current features with stylized activations
        if not final_pass:  # hypercolumn matching / 'hm' regime

            # Extract features from current output, normalize for cos distance
            current_features = extract_features(image_output, phi)
            current_features_n = current_features / get_feat_norms(current_features)

            # Update overall loss w/ cosine loss w.r.t target features
            ell = ell + (1. - (target_features_n * current_features_n).sum(1)).mean()


        else:  # feature splitting / 'fs' regime
            # Extract features from current output (keep each layer seperate 
            # and don't downsample)
            current_features = phi(image_output, final_feature_list, False)

            # Compute matches for each layer. For efficiency don't explicitly 
            # gather matches, only access through distance matrix.
            ell_fs = 0.
            for h_i in range(len(src_features)):
                # Get features from a particular layer
                src_temporary = src_features[h_i]
                current_temporary = current_features[h_i]
                chans = src_temporary.size(1)

                # Sparsely sample feature tensors if too big, otherwise just 
                # reshape
                if max(current_temporary.size(2), current_temporary.size(3)) > 64:
                    stride = max(current_temporary.size(2), current_temporary.size(3)) // 64
                    off_a = random.randint(0, stride - 1)
                    off_b = random.randint(0, stride - 1)
                    src_temporary = src_temporary[:, :, off_a::stride, off_b::stride]
                    view_content_source_temporary = current_temporary[:, :, off_a::stride, off_b::stride]

                r_col_samp = src_temporary.contiguous().view(1, chans, -1)
                s_col_samp = view_content_source_temporary.contiguous().view(1, chans, -1)

                # Compute distance matrix and find minimum along each row to 
                # implicitly get matches (and minimize distance between them)
                distance_matrix = cos_center_distance_pairwise(r_col_samp[0].transpose(1, 0),
                                                      s_col_samp[0].transpose(1, 0))
                d_min, _ = torch.min(distance_matrix, 0)

                # Aggregate loss over layers
                ell_fs = ell_fs + d_min.mean()

            # Update overall loss
            ell = ell + ell_fs

        # Optional self similarity content loss between downsampled output 
        # and content image. Always turn off at end for best results.
        if loss_content and not (final_pass and i > 100):
            o_scl = max(image_output.size(2), image_output.size(3))
            o_fac = o_scl / 32.
            h = int(image_output.size(2) / o_fac)
            w = int(image_output.size(3) / o_fac)

            o_flat = flatten_grid(scl_spatial(image_output, h, w))
            self_simulation_output = distance_l2_pairwise(o_flat, o_flat)

            ell = ell + torch.mean(torch.abs((self_simulation_output - self_sim_target)))

        # Update output's pyramid coefficients
        ell.backward()
        opt.step()

    # Update output's pyramid coefficients for current resolution
    # (and all coarser resolutions)    
    style_pyramid[scl:] = dec_pyr(image_output, len(c_pyr) - 1 - scl)
    return style_pyramid
