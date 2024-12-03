import numpy as np
import cv2
# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)
import matplotlib.pyplot as plt
import torch
import random
import kornia

def generate_random_homography_with_disturbance(pts1, size=(640, 640), max_delta=32):
    """
    为图像的四个角点生成不同的随机扰动，计算单应性矩阵，并生成稠密光流
    :param size: 图像的大小，默认为(256, 256)
    :param max_delta: 最大扰动值，默认为10像素
    :return: 单应性矩阵 H 和生成的稠密光流
    """
    # 原始四个角点的坐标
    # 为每个角点生成独立的随机扰动
    delta = np.random.uniform(-max_delta, max_delta, (4, 2))  # 每个角点的扰动是独立的
    pts2 = pts1 + delta  # 将扰动加到原始角点上

    # 计算单应性矩阵
    H, _ = cv2.findHomography(pts1, pts2)
    
    # 使用网格计算稠密光流
    y_grid, x_grid = np.mgrid[0:size[1], 0:size[0]]  # 创建网格
    points = np.vstack((x_grid.flatten(), y_grid.flatten())).T  # 堆叠x和y坐标

    # 将齐次坐标添加到点上
    ones = np.ones((points.shape[0], 1))
    points_homogeneous = np.hstack([points, ones])  # (N, 3)

    # 应用单应性矩阵变换
    transformed_points = points_homogeneous.dot(H.T)
    transformed_points /= transformed_points[:, 2].reshape(-1, 1)  # 齐次坐标归一化

    # 计算光流：目标位置 - 原始位置
    flow = transformed_points[:, :2] - points  # 计算光流
    flow = flow.reshape(size[1], size[0], 2)  # 将光流重塑成图像大小

    return H, flow, delta


def transformer(I, vgrid, train=True):
    # I: Img, shape: batch_size, 1, full_h, full_w
    # vgrid: vgrid, target->source, shape: batch_size, 2, patch_h, patch_w
    # outsize: (patch_h, patch_w)

    def _interpolate(im, x, y, out_size, scale_h):
        # x: x_grid_flat
        # y: y_grid_flat
        # out_size: same as im.size
        # scale_h: True if normalized
        # constants
        num_batch, num_channels, height, width = im.size()

        out_height, out_width = out_size[0], out_size[1]
        # zero = torch.zeros_like([],dtype='int32')
        zero = 0
        max_y = height - 1
        max_x = width - 1

        # do sampling
        x0 = torch.floor(x).int()
        x1 = x0 + 1
        y0 = torch.floor(y).int()
        y1 = y0 + 1

        x0 = torch.clamp(x0, zero, max_x)  # same as np.clip
        x1 = torch.clamp(x1, zero, max_x)
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)

        dim1 = width * height
        dim2 = width

        if torch.cuda.is_available():
            base = torch.arange(0, num_batch).int().cuda()
        else:
            base = torch.arange(0, num_batch).int()

        base = base * dim1
        base = base.repeat_interleave(out_height * out_width, axis=0)  
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im = im.permute(0, 2, 3, 1).contiguous()
        im_flat = im.reshape([-1, num_channels]).float()

        idx_a = idx_a.unsqueeze(-1).long()
        idx_a = idx_a.expand(out_height * out_width * num_batch, num_channels)
        Ia = torch.gather(im_flat, 0, idx_a)

        idx_b = idx_b.unsqueeze(-1).long()
        idx_b = idx_b.expand(out_height * out_width * num_batch, num_channels)
        Ib = torch.gather(im_flat, 0, idx_b)

        idx_c = idx_c.unsqueeze(-1).long()
        idx_c = idx_c.expand(out_height * out_width * num_batch, num_channels)
        Ic = torch.gather(im_flat, 0, idx_c)

        idx_d = idx_d.unsqueeze(-1).long()
        idx_d = idx_d.expand(out_height * out_width * num_batch, num_channels)
        Id = torch.gather(im_flat, 0, idx_d)

        # and finally calculate interpolated values
        x0_f = x0.float()
        x1_f = x1.float()
        y0_f = y0.float()
        y1_f = y1.float()

        wa = torch.unsqueeze(((x1_f - x) * (y1_f - y)), 1)
        wb = torch.unsqueeze(((x1_f - x) * (y - y0_f)), 1)
        wc = torch.unsqueeze(((x - x0_f) * (y1_f - y)), 1)
        wd = torch.unsqueeze(((x - x0_f) * (y - y0_f)), 1)
        output = wa * Ia + wb * Ib + wc * Ic + wd * Id

        return output

    def _transform(I, vgrid, scale_h):

        C_img = I.shape[1]
        B, C, H, W = vgrid.size()

        x_s_flat = vgrid[:, 0, ...].reshape([-1])
        y_s_flat = vgrid[:, 1, ...].reshape([-1])
        out_size = vgrid.shape[2:]
        input_transformed = _interpolate(I, x_s_flat, y_s_flat, out_size, scale_h)

        output = input_transformed.reshape([B, H, W, C_img])
        return output

    # scale_h = True
    output = _transform(I, vgrid, scale_h=False)
    if train:
        output = output.permute(0, 3, 1, 2).contiguous()
    return output

def get_4cor(flow_gt):
    flow_gt = flow_gt.squeeze(0)
    flow_4cor = torch.zeros((2, 2, 2))
    flow_4cor[:, 0, 0] = flow_gt[:, 0, 0]
    flow_4cor[:, 0, 1] = flow_gt[:, 0, -1]
    flow_4cor[:, 1, 0] = flow_gt[:, -1, 0]
    flow_4cor[:, 1, 1] = flow_gt[:, -1, -1]
    return flow_4cor

def get_warp_flow(img, flow, start=0):
    batch_size, _, patch_size_h, patch_size_w = flow.shape
    grid_warp = get_grid(batch_size, patch_size_h, patch_size_w, start)[:, :2, :, :] + flow
    img_warp = transformer(img, grid_warp)
    return img_warp

def get_grid(batch_size, H, W, start=0):
    '''
    获取网格坐标
    shape: (b,3,h,w)
    item(x,y,z): (x,y,1)
    '''
    if torch.cuda.is_available():
        xx = torch.arange(0, W).cuda()
        yy = torch.arange(0, H).cuda()
    else:
        xx = torch.arange(0, W)
        yy = torch.arange(0, H)
    xx = xx.view(1, -1).repeat(H, 1)
    yy = yy.view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(batch_size, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(batch_size, 1, 1, 1)
    ones = torch.ones_like(xx).cuda() if torch.cuda.is_available() else torch.ones_like(xx)
    grid = torch.cat((xx, yy, ones), 1).float()
    if not isinstance(start,int):
        start=start[:,:,None,None]
    grid[:, :2, :, :] = grid[:, :2, :, :] + start
    return grid

def gen_basis(h, w, is_qr=True, is_scale=True):
    basis_nb = 8
    grid = get_grid(1, h, w).permute(0, 2, 3, 1)  # 1, w, h, (x, y, 1)
    flow = grid[:, :, :, :2] * 0

    names = globals()
    for i in range(1, basis_nb + 1):
        names['basis_' + str(i)] = flow.clone()

    basis_1[:, :, :, 0] += grid[:, :, :, 0]  # [1, w, h, (x, 0)]
    basis_2[:, :, :, 0] += grid[:, :, :, 1]  # [1, w, h, (y, 0)]
    basis_3[:, :, :, 0] += 1  # [1, w, h, (1, 0)]
    basis_4[:, :, :, 1] += grid[:, :, :, 0]  # [1, w, h, (0, x)]
    basis_5[:, :, :, 1] += grid[:, :, :, 1]  # [1, w, h, (0, y)]
    basis_6[:, :, :, 1] += 1  # [1, w, h, (0, 1)]
    basis_7[:, :, :, 0] += grid[:, :, :, 0] ** 2  # [1, w, h, (x^2, xy)]
    basis_7[:, :, :, 1] += grid[:, :, :, 0] * grid[:, :, :, 1]  # [1, w, h, (x^2, xy)]
    basis_8[:, :, :, 0] += grid[:, :, :, 0] * grid[:, :, :, 1]  # [1, w, h, (xy, y^2)]
    basis_8[:, :, :, 1] += grid[:, :, :, 1] ** 2  # [1, w, h, (xy, y^2)]

    flows = torch.cat([names['basis_' + str(i)] for i in range(1, basis_nb + 1)], dim=0)
    if is_qr:
        flows_ = flows.view(basis_nb, -1).permute(1, 0)  # N, h, w, c --> N, h*w*c --> h*w*c, N
        flow_q, _ = torch.linalg.qr(flows_)
        flow_q = flow_q.permute(1, 0).reshape(basis_nb, h, w, 2)
        flows = flow_q

    if is_scale:
        max_value = flows.abs().reshape(8, -1).max(1)[0].reshape(8, 1, 1, 1)
        flows = flows / max_value

    return flows.permute(0, 3, 1, 2)

def warpPerspective(img, H, dsize, mode='bilinear', padding_mode='zeros', align_corners=False):
    """
    封装的 `warpPerspective` 函数，支持 tensor 或 ndarray 输入
    :param img: 输入图像，支持 tensor 或 ndarray
    :param H: 变换矩阵 (3x3)
    :param dsize: 目标输出图像的尺寸 (宽, 高)
    :param mode: 插值方法，'bilinear' 或 'nearest'
    :param padding_mode: 填充模式，'zeros' 或 'border' 等
    :param align_corners: 仅在 `mode='bilinear'` 时有效，是否对齐角点
    :return: 扭曲后的图像，类型与输入类型相同
    """
    if isinstance(img, np.ndarray):  # 如果输入是 ndarray，使用 OpenCV
        # 使用 cv2 的 warpPerspective 进行变换
        return cv2.warpPerspective(img, H, dsize, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    elif isinstance(img, torch.Tensor):  # 如果输入是 tensor，使用 Kornia
        # 使用 Kornia 的 warp_perspective 进行变换
        img = img.unsqueeze(0) if img.ndimension() == 3 else img  # Add batch dimension if needed
        warped_img = kornia.geometry.transform.warp_perspective(img, H, dsize, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
        return warped_img.squeeze(0)  if warped_img.ndimension() == 3 else warped_img # 保持维度一致
    else:
        raise TypeError("Input image must be a numpy ndarray or a torch tensor.")

def resize_and_pad(img, target_height, target_width,keep_ratio=True):
    """调整图像尺寸并填充以适应目标尺寸"""
    if not keep_ratio:
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    original_height, original_width = img.shape[:2]
    scale = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    padded_img = np.zeros((target_height, target_width, img.shape[2]), dtype=img.dtype)
    pad_top = (target_height - new_height) // 2
    pad_left = (target_width - new_width) // 2
    
    padded_img[pad_top:pad_top + new_height, pad_left:pad_left + new_width] = resized_img
    return padded_img

def getDisturbedBox(shape,marginal,patch_size):
    height, width = shape
    y = random.randint(marginal, height - marginal - patch_size)
    x = random.randint(marginal, width - marginal - patch_size)
    top_left_point = (x, y)
    bottom_right_point = (patch_size + x, patch_size + y)


    top_left_point_new = (x, y)
    bottom_left_point_new = (x, patch_size + y - 1)
    bottom_right_point_new = (patch_size + x - 1, patch_size + y - 1)
    top_right_point_new = (x + patch_size - 1, y)
    four_points_new = [top_left_point_new, top_right_point_new, bottom_left_point_new, bottom_right_point_new]


    perturbed_four_points_new = []
    for i in range(4):
                
        t1 = random.randint(-marginal, marginal)
        t2 = random.randint(-marginal, marginal)

        perturbed_four_points_new.append((four_points_new[i][0] + t1,
                                            four_points_new[i][1] + t2))
        
    crop_fn = lambda img: img[top_left_point[1]:bottom_right_point[1], 
                              top_left_point[0]:bottom_right_point[0]]
    return four_points_new,perturbed_four_points_new,crop_fn


def getFlow(shape,H,H_inverse):
    y_grid, x_grid = np.mgrid[0:shape[0], 0:shape[1]]
    point = np.vstack((x_grid.flatten(), y_grid.flatten())).transpose()

    point_transformed_branch1 = cv2.perspectiveTransform(np.array([point], dtype=np.float64), H).squeeze()
    point_transformed_branch1_inv = cv2.perspectiveTransform(np.array([point], dtype=np.float64), H_inverse).squeeze()

    diff_branch = point_transformed_branch1 - np.array(point, dtype=np.float64)
    diff_branch_inv = point_transformed_branch1_inv - np.array(point, dtype=np.float64)
    return diff_branch.reshape(shape[0], shape[1],-1),diff_branch_inv.reshape(shape[0], shape[1],-1)

def getFlowWithTorch(shape,H,H_inverse):
    """
    用 Kornia 实现计算光流的函数。
    
    Args:
        shape (tuple): 图像的形状，(height, width)。
        H (torch.Tensor): 单应性变换矩阵，形状为 (3, 3)，需为 (1, 3, 3) 批次格式。
        H_inverse (torch.Tensor): 单应性变换的逆矩阵，形状为 (3, 3)，需为 (1, 3, 3) 批次格式。
    
    Returns:
        diff_branch (torch.Tensor): 正向单应性变换的光流，形状为 (height, width, 2)。
        diff_branch_inv (torch.Tensor): 逆向单应性变换的光流，形状为 (height, width, 2)。
    """
    H=torch.from_numpy(H)
    H_inverse=torch.from_numpy(H_inverse)
    # 图像的高度和宽度
    height, width = shape

    # 创建网格
    y_grid, x_grid = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    grid = torch.stack([x_grid, y_grid], dim=-1) # (height, width, 2)
    
    # 转换为齐次坐标 (batch_size=1, num_points, 3)
    grid_h = kornia.geometry.linalg.convert_points_to_homogeneous(grid.view(-1, 2))  # (H*W, 3)
    grid_h = grid_h.unsqueeze(0)  # 添加批次维度 (1, H*W, 3)

    # 正向单应性变换
    point_transformed_branch1 = kornia.geometry.transform_points(H.unsqueeze(0), grid_h)  # (1, H*W, 3)
    point_transformed_branch1 = point_transformed_branch1.squeeze(0)[:, :2]  # 去掉批次维度，提取 (H*W, 2)

    # 逆向单应性变换
    point_transformed_branch1_inv = kornia.geometry.transform_points(H_inverse.unsqueeze(0), grid_h)
    point_transformed_branch1_inv = point_transformed_branch1_inv.squeeze(0)[:, :2]

    # 计算光流差值
    diff_branch = point_transformed_branch1 - grid.view(-1, 2)
    diff_branch_inv = point_transformed_branch1_inv - grid.view(-1, 2)

    # 调整为图像形状 (height, width, 2)
    diff_branch = diff_branch.view(height, width, 2)
    diff_branch_inv = diff_branch_inv.view(height, width, 2)

    return diff_branch, diff_branch_inv

def generateTrainImagePair(img1, img2, marginal=32, patch_size=640,reshape=None,keep_ratio=True):
    
    # 检查并调整图像
    min_height = 2 * marginal + patch_size
    min_width = 2 * marginal + patch_size
    if reshape is not None:
        min_height=reshape[0]
        min_width=reshape[1]
        img1 = resize_and_pad(img1, min_height, min_width)
        img2 = resize_and_pad(img2, min_height, min_width)
    elif img1.shape[0] < min_height or img1.shape[1] < min_width:
        img1 = resize_and_pad(img1, min_height, min_width)
        img2 = resize_and_pad(img2, min_height, min_width)
    
    four_points_new,perturbed_four_points_new, crop_fn = getDisturbedBox(img1.shape[0:-1], marginal, patch_size)

    org_corners = np.float32(four_points_new)
    dst = np.float32(perturbed_four_points_new)
    disturbed_corners = (dst-org_corners).reshape(2,2,2) # w,h,(x,y)
    H = cv2.getPerspectiveTransform(org_corners, dst)
    H_inverse = np.linalg.inv(H)

    warped_image = warpPerspective(img2, H_inverse, (img1.shape[1], img1.shape[0]))

    img_patch_ori = crop_fn(img1)
    img_patch_pert = crop_fn(warped_image)

    # diff_branch,diff_branch_inv=getFlow(img1.shape[0:2],H,H_inverse)
    # pf_patch=crop_fn(diff_branch)
    # pf_patch_inv=crop_fn(diff_branch_inv)

    pf_patch,pf_patch_inv=1,1

    return {
            "img1_patch": img_patch_ori,
            "img2_patch": img_patch_pert,
            "img1_full": img1,
            "img2_full": warped_image,
            "flow_img1": pf_patch,
            "flow_img2": pf_patch_inv,
            "disturbed_corners": disturbed_corners,
            "origin_corners": org_corners,
            "H": H,
        }



if __name__=='__main__':
    print(get_grid(1,320,640).shape)