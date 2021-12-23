"""
An implementation of II2S, an improved version of Image2StyleGAN, 
as claimed in reference paper:
{Peihao Zhu, Rameen Abdal, Yipeng Qin, John Femiani, and Peter Wonka
.2020b.Improved StyleGAN Embedding: Where are the Good Latents?
arXiv:2012.09036 [cs.CV]}
"""

import argparse
import os

import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import sys
sys.path.append('stylegan2_pytorch')
import lpips
from model import Generator
from utils import *

latent_savefile='latent_stat'
svd_savefile='svd_S_V'

def prepare_latent_data(g_ema, latent_savefile, svd_savefile, n_mean_latent = 10**6, P_savefile=None):
    CHUNK = 10**5
    count = n_mean_latent
    torch.manual_seed(1)
    latent_out_list = []
    with torch.no_grad():
        while count > 0:
            print(count)
            noise_sample = torch.randn(CHUNK, 512, device='cuda')
            latent_out = g_ema.style(noise_sample)
            latent_out_list.append(latent_out)
            count -= CHUNK
        latent_out = torch.vstack(latent_out_list)
        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / (n_mean_latent - 1) ) ** 0.5

    latent_stat = {'mean': latent_mean, 'std': latent_std}
    torch.save(latent_stat, latent_savefile)
    print("latent static data saved in:", latent_savefile)
    # use P in accordance with II2S paper
    P = F.leaky_relu(latent_out, negative_slope=5)
    if P_savefile is not None:
        P_dict = {'P': P}
        torch.save(P_dict, P_savefile)
        print("P data saved in:", P_savefile)

    P_mean = P.mean(0)
    _, S, V = torch.svd(P - P_mean)

    svd_dict = {'S':S, 'V':V, 'mean': P_mean}
    torch.save(svd_dict, svd_savefile)
    print("svd result saved in:", svd_savefile)

def II2S(args, save=True, device="cuda", resize=512):
    """ Project input image(s) to latent code(s)"""
    resize = min(args.size, resize)

    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    imgs = []

    for imgfile in args.files:
        img = transform(Image.open(imgfile).convert("RGB"))
        imgs.append(img)

    imgs = torch.stack(imgs, 0).to(device)

    g_ema = Generator(args.size, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)

    percept = lpips.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
    )

    if (os.path.exists(latent_savefile) and os.path.exists(svd_savefile)) == False:
        prepare_latent_data(g_ema, latent_savefile, svd_savefile)
    latent_stat = torch.load(latent_savefile)
    latent_mean, latent_std = latent_stat['mean'], latent_stat['std']
    svd_dict = torch.load(svd_savefile)
    S, V, P_mean = svd_dict['S'], svd_dict['V'], svd_dict['mean']
    S_inv = 1 / S

    noises_single = g_ema.make_noise()
    noises = []
    for noise in noises_single:
        noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())

    latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1)

    if args.w_plus:
        latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)

    latent_in.requires_grad = True

    for noise in noises:
        noise.requires_grad = False

    optimizer = optim.Adam([latent_in] + noises, lr=args.lr)

    pbar = tqdm(range(args.step))
    latent_path = []
    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr
        if args.noise > 0:
            noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
            latent_n = latent_noise(latent_in, noise_strength.item())
        else:
            latent_n = latent_in
        img_gen, _ = g_ema([latent_n], input_is_latent=True, noise=noises)

        batch, channel, height, width = img_gen.shape

        if height > resize:
            factor = height // resize

            img_gen = img_gen.reshape(
                batch, channel, height // factor, factor, width // factor, factor
            )
            img_gen = img_gen.mean([3, 5])

        p_loss = percept(img_gen, imgs).sum()
        
        # n_loss = noise_regularize(noises)
        n_loss = torch.Tensor([0]).cuda()

        mse_loss = F.mse_loss(img_gen, imgs)

        pn_loss = PN_loss(latent_in, P_mean, V, S_inv)

        loss = p_loss + args.noise_regularize * n_loss + args.mse * mse_loss + args.pn * pn_loss


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        noise_normalize_(noises)

        if (i + 1) % 100 == 0:
            latent_path.append(latent_in.detach().clone())

        pbar.set_description(
            (
                f"perceptual: {p_loss.item():.4f}; noise regularize: {n_loss.item():.4f};"
                f" mse: {mse_loss.item():.4f}; lr: {lr:.6f}"
            )
        )

    if save:
        img_gen, _ = g_ema([latent_path[-1]], input_is_latent=True, noise=noises)

        filename = os.path.splitext(os.path.basename(args.files[0]))[0] + ".pt"

        img_ar = make_image(img_gen)

        result_file = {}
        for i, input_name in enumerate(args.files):
            noise_single = []
            for noise in noises:
                noise_single.append(noise[i : i + 1])

            result_file[input_name] = {
                "img": img_gen[i],
                "latent": latent_in[i],
                "noise": noise_single,
            }

            img_name = os.path.splitext(os.path.basename(input_name))[0] + "-project.png"
            pil_img = Image.fromarray(img_ar[i])
            pil_img.save(os.path.join(args.out_dir, img_name))

        torch.save(result_file, os.path.join(args.out_dir, filename))

    return latent_path[-1]

if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(
        description="Image projector to the generator latent spaces"
    )
    parser.add_argument(
        "--ckpt", type=str, required=True, help="path to the model checkpoint"
    )
    parser.add_argument(
        "--size", type=int, default=256, help="output image sizes of the generator"
    )
    parser.add_argument(
        "--lr_rampup",
        type=float,
        default=0.05,
        help="duration of the learning rate warmup",
    )
    parser.add_argument(
        "--lr_rampdown",
        type=float,
        default=0.25,
        help="duration of the learning rate decay",
    )
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument(
        "--noise", type=float, default=0.0, help="strength of the noise level"
    )
    parser.add_argument(
        "--noise_ramp",
        type=float,
        default=0.75,
        help="duration of the noise level decay",
    )
    parser.add_argument("--step", type=int, default=1300, help="optimize iterations")
    parser.add_argument(
        "--noise_regularize",
        type=float,
        default=1e5,
        help="weight of the noise regularization",
    )
    parser.add_argument(
        "--pn",
        type=float,
        default=0.001,
        help='weight of the PN+ space regularization'
    )
    parser.add_argument(
        "--mse", 
        type=float, 
        default=1, 
        help="weight of the mse loss")
    parser.add_argument(
        "--w_plus",
        action="store_true",
        help="allow to use distinct latent codes to each layers",
    )
    parser.add_argument(
        "files", metavar="FILES", nargs="+", help="path to image files to be projected"
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default='out'
    )
    args = parser.parse_args()

    II2S(args, device=device)