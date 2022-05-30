import torch

def whiten(x, transpose_eig_vec_of_whitening_cov, u, s):
    '''
    Applies whitening as described in:
    https://openaccess.thecvf.com/content_ICCV_2019/papers/Chiu_Understanding_Generalized_Whitening_and_Coloring_Transform_for_Universal_Style_Transfer_ICCV_2019_paper.pdf
    x -- N x D pytorch tensor
    transpose_eig_vec_of_whitening_cov -- D x D transposed eigenvectors of whitening covariance
    u  -- D x D eigenvectors of whitening covariance
    s  -- D x 1 eigenvalues of whitening covariance
    '''
    tps = lambda x: x.transpose(1, 0)
    return tps(torch.matmul(u, torch.matmul(transpose_eig_vec_of_whitening_cov, tps(x)) / s))

def colorize(x, transpose_eig_vec_of_whitening_cov, u, s):
    '''
    Applies "coloring transform" as described in:
    https://openaccess.thecvf.com/content_ICCV_2019/papers/Chiu_Understanding_Generalized_Whitening_and_Coloring_Transform_for_Universal_Style_Transfer_ICCV_2019_paper.pdf
    x -- N x D pytorch tensor
    transpose_eig_vec_of_whitening_cov -- D x D transposed eigenvectors of coloring covariance
    u  -- D x D eigenvectors of coloring covariance
    s  -- D x 1 eigenvalues of coloring covariance
    '''
    tps = lambda x: x.transpose(1, 0)
    return tps(torch.matmul(u, torch.matmul(transpose_eig_vec_of_whitening_cov, tps(x)) * s))

def zca(content, style):
    '''
    Matches the mean and covariance of 'content' to those of 'style'
    content -- N x D pytorch tensor of content feature vectors
    style   -- N x D pytorch tensor of style feature vectors
    '''
    mu_c = content.mean(0, keepdim=True)
    mu_s = style.mean(0, keepdim=True)

    content = content - mu_c
    style = style - mu_s

    convolution_for_content = torch.matmul(content.transpose(1,0), content) / float(content.size(0))
    convolution_for_style = torch.matmul(style.transpose(1,0), style) / float(style.size(0))

    u_c, sigmoid_content, _ = torch.svd(convolution_for_content + torch.eye(convolution_for_content.size(0)).cuda()*1e-4)
    u_s, sigmoid_style, _ = torch.svd(convolution_for_style + torch.eye(convolution_for_style.size(0)).cuda()*1e-4)

    sigmoid_content = sigmoid_content.unsqueeze(1)
    sigmoid_style = sigmoid_style.unsqueeze(1)


    u_c_i = u_c.transpose(1,0)
    u_s_i = u_s.transpose(1,0)

    scl_c = torch.sqrt(torch.clamp(sigmoid_content, 1e-8, 1e8))
    scl_s = torch.sqrt(torch.clamp(sigmoid_style, 1e-8, 1e8))


    whitening_content = whiten(content, u_c_i, u_c, scl_c)
    colorizing_content = colorize(whitening_content, u_s_i, u_s, scl_s) + mu_s

    return colorizing_content, convolution_for_style

def zca_tensor(content, style):
    '''
    Matches the mean and covariance of 'content' to those of 'style'
    content -- B x D x H x W pytorch tensor of content feature vectors
    style   -- B x D x H x W pytorch tensor of style feature vectors
    '''
    content_rs_permute = content.permute(0,2,3,1).contiguous().view(-1,content.size(1))
    style_rs_permute = style.permute(0,2,3,1).contiguous().view(-1,style.size(1))

    view_cs, convolution_for_style = zca(content_rs_permute, style_rs_permute)

    view_cs = view_cs.view(content.size(0),content.size(2),content.size(3),content.size(1)).permute(0,3,1,2)
    return view_cs.contiguous(), convolution_for_style
