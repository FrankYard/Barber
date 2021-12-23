import torch
from torch import Tensor
from torch.optim import Adam
from typing import Callable, List
import math
import cv2
import numpy as np

def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise


def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to("cpu")
        .numpy()
    )

def PN_loss(latent_in, P_mean, V, S_inv):
    x = torch.nn.functional.leaky_relu(latent_in, negative_slope=5)
    v = torch.matmul(x - P_mean, V) * S_inv
    return (v ** 2).mean()

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='out/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    vis_parsing_maps.part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = vis_parsing_maps.part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # return vis_im

def make_seg_mask(parsing, seg_list):
    """
        parsing.shape == (c,h,w)
    """
    seg_mask = None
    for segid in seg_list:
        keep = (parsing == segid)
        if seg_mask is None:
            seg_mask = keep
        else:
            seg_mask.logical_or_(keep)
    return seg_mask

def make_mask(segout, ks, return_seg_mask=False):
    """
        parsings.shape == (b, c, h, w)
    """
    parsings = segout.argmax(1)
    parsing_merged = torch.zeros_like(parsings)
    seg_masks = []
    for i, seg_list in enumerate(ks):
        mask_keep = make_seg_mask(parsings[i], seg_list)
        seg_masks.append(mask_keep)
        parsing_merged[i] = torch.where(mask_keep, parsings[i], 0) #the parses which are not specified in ks are set to 0
    parsing_merged, index  = parsing_merged.max(0)
    if return_seg_mask:
        return parsings, parsing_merged, seg_masks
    else:
        return parsings, parsing_merged

def seg2onehot(parsing, n_class):
    one_hot = torch.zeros((n_class,) + parsing.shape, device=parsing.device)
    for i in range(n_class):
        mask_set = parsing == i
        one_hot[i:, mask_set] = 1
    return one_hot

def resize_tensor(tensor, resize):
    batch, channel, height, width = tensor.shape
    if height > resize:
        factor = height // resize

        tensor = tensor.reshape(
            batch, channel, height // factor, factor, width // factor, factor
        )
        tensor = tensor.mean([3, 5])
    return tensor

def get_val_dict(data_files, input_name):
    for file in data_files:
        data_dict = torch.load(file)
        for key in data_dict.keys():
            if key.find(input_name) != -1:
                return file, data_dict, data_dict[key]
    return None

def adam_proj(params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor],
         state_steps: List[int],
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: float,
         weight_decay: float,
         eps: float,
         projector: Callable):

    for i, param in enumerate(params):

        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]
        if amsgrad:
            max_exp_avg_sq = max_exp_avg_sqs[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        update = exp_avg / denom * step_size
        if projector is not None:
            projector(update, i)
        param.subtract_(update)

class AdamProj(Adam):
    def __init__(self, params, project_vectors, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        self.project_vectors = []
        for i, param in enumerate(params):
            assert param.shape == project_vectors[i].shape
            self.project_vectors.append(project_vectors[i] / torch.norm(project_vectors[i], dim=0, keepdim=True))

    def projector(self, update, i):
        component = (update * self.project_vectors[i]).sum(0, keepdim=True) * self.project_vectors[i]
        update.subtract_(component)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_sums = []
            max_exp_avg_sqs = []
            state_steps = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            beta1, beta2 = group['betas']
            adam_proj(params_with_grad,
                   grads,
                   exp_avgs,
                   exp_avg_sqs,
                   max_exp_avg_sqs,
                   state_steps,
                   group['amsgrad'],
                   beta1,
                   beta2,
                   group['lr'],
                   group['weight_decay'],
                   group['eps'],
                   projector=self.projector
                   )
        return loss
