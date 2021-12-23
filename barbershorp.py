"""
An implemetantion of hair style replacement method method from paper
{Peihao Zhu, Rameen Abdal, John Femiani and Peter Wonka
.2021. Barbershop: gan-based image compositing using segmentation masks.
arXiv:2106.01505 [cs.CV]}
"""

import os
from scipy.optimize.optimize import bracket

import torch
from torch import optim
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from pytorchcv.model_provider import get_model as ptcv_get_model
from PIL import Image
from tqdm import tqdm
from cv2 import cv2 # for visualization on Windows system

import sys
sys.path.append('stylegan2_pytorch')
import lpips
from lpips.networks_basic import VGG16StyleLoss
from model import Generator
from utils import *
import config


transform = transforms.Compose(
    [
        transforms.Resize(config.resize),
        transforms.CenterCrop(config.resize),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)
bisenet = ptcv_get_model("bisenet_resnet18_celebamaskhq", pretrained=True, in_size=(config.resize, config.resize),)
bisenet.cuda()
bisenet.eval()

images = []
imgs = []
for imgfile in (config.files):
    image = Image.open(imgfile).convert("RGB")
    img = transform(image).to(config.device)
    image = image.resize((512, 512), Image.BILINEAR)
    imgs.append(img)
    images.append(image)
imgs = torch.stack(imgs, 0)

"""get image masks using bisenet"""
with torch.no_grad():
    out = bisenet(imgs)[0] # (b, c, h, w)
    segs, merged_seg, seg_masks = make_mask(out, config.parse_ids, return_seg_mask=True)
    merged_seg_onehot = seg2onehot(merged_seg, 19) # withot batch dim

    parsing = merged_seg.cpu().numpy()
    vis_file = os.path.splitext(os.path.basename(config.files[0]))[0] + '-merged_mask.jpg'
    vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=vis_file)
if config.vis_parsing:
    for i_img, image in enumerate(images):
        parsing = out[i_img].cpu().numpy().argmax(0)
        print(np.unique(parsing))
        
        vis_file = os.path.splitext(os.path.basename(config.files[i_img]))[0] + '-mask.jpg'
        vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=vis_file)

"""load image generator based on StyleGAN"""
g_ema = Generator(config.size, 512, 8)
g_ema.load_state_dict(torch.load(config.ckpt)["g_ema"], strict=False)
g_ema.eval()
g_ema = g_ema.to(config.device)

"""load loss function"""
percept = lpips.PerceptualLoss(
    model="net-lin", net="vgg", use_gpu=config.device.startswith("cuda")
)
style_loss = VGG16StyleLoss()
cross_entropy_loss = CrossEntropyLoss(ignore_index=-1)

"""prepare data"""
F_blend = torch.zeros((1,512,32,32), device=config.device)
skip_blend = torch.zeros((1,3,32,32), device=config.device)
latent_aligns = []
img_references = []
merged_seg_masks = []
init_val_keys = ['img', 'latent', 'noise']
if not os.path.exists(config.save_file):
    # When blend these imgs for the first time
    for i_dic, input_name in enumerate(config.files):
        img = imgs[i_dic:i_dic+1]
        data_file, data_dict, val_dict = get_val_dict(config.data_files, input_name)
        if config.load_backup and 'latent_rec' in val_dict.keys():
            latent = val_dict['latent_rec'].clone().detach() # appearance code with non-appearance part
            F_rec = val_dict['F_rec'].clone().detach() # structure tensor
            noise_single = val_dict['noise']
            latent.requires_grad = True
        else:
            """---get structure tensor and appearance code---"""
            _, latent_init, noise_single = [val_dict[val_key] for val_key in init_val_keys] # noise_single is a list

            F_ini, skip = g_ema.get_F(latent_init[None], noise_single)

            latent = latent_init.detach()[None] #(18, 512) -> (1, 18, 512)
            latent.requires_grad = True
            F_ini = F_ini.detach()
            F_rec = F_ini.clone()
            F_rec.requires_grad = True
            skip = skip.detach()
            for noise in noise_single:
                noise.requires_grad = False

            optimizer = optim.Adam([F_rec, latent] + noise_single, lr=config.lr)

            pbar = tqdm(range(config.step))
            for i in pbar:
                t = i / config.step
                lr = get_lr(t, config.lr)
                optimizer.param_groups[0]["lr"] = lr

                img_gen = g_ema.F2img(latent, F_rec, skip, noise_single)
                img_gen = resize_tensor(img_gen, config.resize)

                p_loss = percept(img_gen, img).sum()
                f_loss = ((F_rec - F_ini)**2).mean()
                loss = p_loss + f_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_description(
                    (
                        f"perceptual: {p_loss.item():.4f};"
                        f" F_loss: {f_loss.item():.4f}; lr: {lr:.6f}"
                    )
                )


            if config.save:
                with torch.no_grad():
                    img_gen = g_ema.F2img(latent, F_rec, skip, noise_single)
                img_rec = make_image(img_gen)
                img_rec_name = os.path.splitext(os.path.basename(input_name))[0] + "-rec.png"
                pil_img = Image.fromarray(img_rec[0])
                pil_img.save(img_rec_name)

                val_dict['latent_rec'] = latent.clone().detach() # the non-appearance part is kept for code simplicity
                val_dict['F_rec'] = F_rec.clone().detach()
                val_dict['img_rec'] = img_rec[0]
                torch.save(data_dict, data_file)
        
        """----- Alignment----"""
        size_l8 = F_rec.shape[-2:]
        optimizer = optim.Adam([latent], lr=config.lr_align)

        pbar = tqdm(range(config.step_align))
        for i in pbar:
            t = i / config.step_align
            lr = get_lr(t, config.lr_align)
            optimizer.param_groups[0]["lr"] = lr

            img_gen, _ = g_ema([latent], input_is_latent=True, noise=noise_single)
            img_gen = resize_tensor(img_gen, config.resize)

            seg_gen = bisenet(img_gen)[0]
            seg_mask = make_seg_mask(seg_gen.argmax(1), config.parse_ids[i_dic])
            cv2.imshow('1', seg_mask[0].cpu().float().numpy())
            cv2.waitKey(1)
            cv2.imshow('2', seg_masks[i_dic].cpu().float().numpy())
            cv2.waitKey(1)
            cv2.imshow('3', make_image(img_gen.clone())[0][:,:, [2,1,0]])
            cv2.waitKey(1)
            merged_seg_mask = make_seg_mask(merged_seg, config.parse_ids[i_dic])
            cv2.imshow('4', merged_seg_mask.cpu().float().numpy())
            cv2.waitKey(1)            
            s_loss = style_loss(img_gen, img, seg_mask1=seg_mask, seg_mask2=seg_masks[i_dic])
            # s_loss = style_loss(img_gen, img, seg_mask1=seg_mask, seg_mask2=merged_seg_mask)
            
            ce_loss = cross_entropy_loss(seg_gen, merged_seg[None])
            # ce_merged_seg = torch.where(seg_masks[i_dic], -1, merged_seg[None])
            # ce_loss = cross_entropy_loss(seg_gen, ce_merged_seg)

            ce_loss = ce_loss.mean()
            loss = ce_loss + 15000.0 * s_loss
            # loss = 15000.0 * ce_loss + s_loss
            # loss = ce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(
                (
                    f"style loss: {s_loss.item():.4f};"
                    f" ce loss: {ce_loss.item():.8f}; lr: {lr:.6f}"
                )
            )

        """--- Structure Transfer---"""
        # optimizer.zero_grad(set_to_none=True)
        # torch.cuda.empty_cache()
        with torch.no_grad():
            F_tem, skip_tem = g_ema.get_F(latent, noise_single)
            img_reference = g_ema.F2img(latent, F_tem, skip_tem, noise_single)

            merged_seg_mask = make_seg_mask(merged_seg, config.parse_ids[i_dic])[None,None].float()
            merged_seg_mask_l8 = F.interpolate(merged_seg_mask, size_l8, mode='bicubic')
            seg_mask_l8 = F.interpolate(seg_masks[i_dic][None,None].float(), size_l8, mode='bicubic')
            mask_rec = merged_seg_mask_l8 * seg_mask_l8
            F_align = mask_rec * F_rec + (1 - mask_rec) * F_tem

            val_dict['latent_align'] = latent
            val_dict['F_align'] = F_align
            latent_aligns.append(latent)
            img_references.append(img_reference)
            merged_seg_masks.append(merged_seg_mask)

            imshow_align = make_image(F_align[:1,:3].clone())[0][:,:,[2,1,0]]
            imshow_align = cv2.resize(imshow_align, (100,100))
            cv2.imshow('F_align_{}'.format(i_dic), imshow_align)
            cv2.waitKey(1)
            cv2.imshow('img_reference', make_image(img_reference.clone())[0][:,:, [2,1,0]])
            cv2.waitKey(1)
        
            """----Strusture Blending----"""
            merged_mask_blend = make_seg_mask(merged_seg, config.parse_ids[i_dic])[None,None].float()
            merged_mask_blend_l8 = F.interpolate(merged_mask_blend, size_l8, mode='bicubic')
            F_blend += merged_mask_blend_l8 * F_align
            skip_blend += merged_mask_blend_l8 * skip_tem
            if config.save:
                torch.save(data_dict, data_file)
    "end for"

    """---- Appearance Blending----"""
    # torch.cuda.empty_cache()
    latent_aligns = torch.stack(latent_aligns, dim=0).detach()
    u = torch.zeros_like(latent_aligns)
    # u[0,0,-1] = 1
    u[-1] = 1
    latent_aligns.requires_grad = False
    u.requires_grad = True


    optimizer = AdamProj([u], lr=config.lr_blend, project_vectors=[torch.ones_like(u)])
    pbar = tqdm(range(config.step))
    for i in pbar:
        t = i / config.step
        lr = get_lr(t, config.lr_blend)
        optimizer.param_groups[0]["lr"] = lr
        latent_blend = (latent_aligns * u).sum(0)

        img_gen = g_ema.F2img(latent_blend, F_blend, skip_blend, noise_single)
        img_gen = resize_tensor(img_gen, config.resize)
        cv2.imshow('img_blending', make_image(img_gen.clone())[0][:,:, [2,1,0]])
        cv2.waitKey(1)

        loss = 0
        for i, img_ref in enumerate(img_references):
            img_ref = resize_tensor(img_ref, config.resize)
            loss += percept.model.net.forward(img_gen ,img_ref, mask=merged_seg_masks[i])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    save_dict = {
        'latent_blend':latent_blend,
        'F_blend':F_blend,
        'skip_blend':skip_blend,
        'noise_single': noise_single}
    torch.save(save_dict, config.save_file)
else:
    # when these imgs are already blended
    print('blend result found, loading...')
    save_dict = torch.load(config.save_file)
    latent_blend, F_blend, skip_blend, noise_single = save_dict['result'].values()
    # cv2.imshow('skip',make_image(skip_blend.clone())[0][:,:,[2,1,0]])
    cv2.imshow('F_blend[?]',make_image(F_blend[:1,:3].clone())[0][:,:,[2,1,0]])
    cv2.waitKey(1)
with torch.no_grad():
    # latent_blend = list(init_dict.values())[1]['latent'][None] # use unoptimized latent
    img_gen = g_ema.F2img(latent_blend, F_blend, skip_blend, noise_single)
    img_blend = make_image(img_gen)
    img_blend_name = os.path.splitext(os.path.basename(data_file))[0] + "-ble.png"
    pil_img = Image.fromarray(img_blend[0])
    cv2.imshow('4', img_blend[0][:,:, [2,1,0]])
    cv2.waitKey(1)
    print('saving result:', img_blend_name)
    pil_img.save(img_blend_name)
