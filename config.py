"""
---config for barbershorp.py---
"""
"""
files: image paths in accordance with II2S.
parse_ids: controls how images in files are blended:
parse_ids[i] are sematic ids that control which part of 
files[i] to be blended into the result image.
Require len(files) == len(parse_ids).
"""

files = ['ImagePath1.png', 'ImagePath2.png', 'ImagePath1.png'] # It's OK to enter the same img path
parse_ids = [[0], [17], list(range(1, 17)) + [18,19]] # 0->background, 17->hair
data_files = ['out\\ImagePath1.pt', 'out\\ImagePath2.pt'] # files to restore images' latent codes
save_file = 'out\\blended_result' # file to save blended result code
ckpt = "stylegan2-ffhq-config-f.pt" # StyleGAN weight path

"""some parameters"""
resize = 512
size = 1024
lr = 0.01
lr_align = 0.0001
lr_blend = 0.01
step = 100 # setp for structure tensor building and appearance blending
step_align = 100
device = "cuda"
load_backup = True
save = True
vis_parsing = False