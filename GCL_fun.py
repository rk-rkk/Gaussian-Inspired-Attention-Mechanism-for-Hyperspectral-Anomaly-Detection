import numpy as np
import scipy.io
import torch
import torch.optim

from AUC_cal import AUC_cal
from models.modelsSkip import skip
from utlis.commonUtils import get_noise, get_params

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor


def GCL_main(data_path):
    # data input
    # **************************************************************************************************************
    torch.cuda.empty_cache()
    root_path = data_path
    thres = 0.000015
    channellss = 128
    layers = 5

    file_name = root_path + ".mat"
    # mat = h5py.File(file_name)
    mat = scipy.io.loadmat(file_name)
    img_h5 = mat["data"]
    img_np = np.array(img_h5)

    min_value = img_np.min()
    max_value = img_np.max()
    img_np = (img_np - min_value) / (max_value - min_value)

    img_np = img_np.transpose(2, 0, 1)
    img_np = img_np.astype(np.float32)
    img_var = torch.from_numpy(img_np).type(dtype)
    img_size = img_var.size()
    band = img_size[0]
    row = img_size[1]
    col = img_size[2]

    # model setup
    # **************************************************************************************************************
    pad = 'reflection'  # 'zero'
    OPT_OVER = 'net'
    # OPTIMIZER = 'adam'
    method = '2D'
    input_depth = img_np.shape[0]
    LR = 0.01
    num_iter = 1001
    param_noise = False
    reg_noise_std = 0.1  # 0 0.01 0.03 0.05
    net = skip(input_depth, img_np.shape[0], num_channels_down=[channellss] * layers,
               num_channels_up=[channellss] * layers,
               num_channels_skip=[channellss] * layers,
               filter_size_up=3, filter_size_down=3,
               upsample_mode='nearest', filter_skip_size=1,
               need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)
    net = net.type(dtype)  # see network structure

    torch.manual_seed(42)  # 为CPU设置随机种子
    torch.cuda.manual_seed(42)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(42)  # 为所有GPU设置随机种子

    net_input = get_noise(input_depth, method, img_np.shape[1:]).type(dtype)

    # s = sum(np.prod(list(p.size())) for p in net.parameters())
    # print('Number of params: %d' % s)

    # Loss
    mse = torch.nn.MSELoss().type(dtype)
    img_var = img_var[None, :].cuda()

    mask_var = torch.ones(1, band, row, col).cuda()
    residual_varr = torch.ones(row, col).cuda()

    def closure(iter_num, mask_varr, residual_varr):

        if param_noise:
            for n in [x for x in net.parameters() if len(x.size()) == 4]:
                n = n + n.detach().clone().normal_() * n.std() / 50

        net_input = net_input_saved
        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        out = net(net_input)
        out_np = out.detach().cpu().squeeze().numpy()

        mask_var_clone = mask_varr.detach().clone()
        residual_var_clone = residual_varr.detach().clone()

        if iter_num % 100 == 0 and iter_num != 0:
            # weighting block
            img_var_clone = img_var.detach().clone()
            net_output_clone = out.detach().clone()
            temp = (net_output_clone[0, :] - img_var_clone[0, :]) * (net_output_clone[0, :] - img_var_clone[0, :])
            residual_img = temp.sum(0)

            residual_var_clone = residual_img
            r_max = residual_img.max()
            # residuals to weights
            residual_img = r_max - residual_img
            r_min, r_max = residual_img.min(), residual_img.max()
            residual_img = (residual_img - r_min) / (r_max - r_min)

            mask_size = mask_var_clone.size()
            for i in range(mask_size[1]):
                mask_var_clone[0, i, :] = residual_img[:]

        total_loss = mse(out * mask_var_clone, img_var * mask_var_clone)
        total_loss.backward()

        return mask_var_clone, residual_var_clone, out_np, total_loss

    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    loss_np = np.zeros((1, 50), dtype=np.float32)
    loss_last = 0
    end_iter = False
    p = get_params(OPT_OVER, net, net_input)

    optimizer = torch.optim.Adam(p, lr=LR)
    for j in range(num_iter):
        optimizer.zero_grad()
        mask_var, residual_varr, background_img, loss = closure(j, mask_var, residual_varr)
        optimizer.step()

        if j >= 1:
            index = j - int(j / 50) * 50
            loss_np[0][index - 1] = abs(loss - loss_last)
            if j % 50 == 0:
                mean_loss = np.mean(loss_np)
                if mean_loss < thres:
                    end_iter = True

        loss_last = loss

        if j == num_iter - 1 or end_iter == True:
            residual_np = residual_varr.detach().cpu().squeeze().numpy()
            auc = AUC_cal(mat["map"], residual_np)
            return auc, loss, end_iter, residual_np


if __name__ == "__main__":
    GCL_main("HYDICE")
