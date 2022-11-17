import matplotlib.pyplot as plt
from matplotlib import animation

import cv2
from PIL import Image
import numpy as np
import importlib
import os
import argparse
from tqdm import tqdm
import torch

from E2FGVI.core.utils import to_tensors



w, h = 432, 240
ref_length = 10  # ref_step
num_ref = -1
neighbor_stride = 5


# sample reference frames from the whole video 
def get_ref_index(f, neighbor_ids, start, length):
    ref_index = []
    if num_ref == -1:
        for i in range(start, length, ref_length):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(start, f - ref_length * (num_ref//2))
        end_idx = min(length, f + ref_length * (num_ref//2))
        for i in range(start_idx, end_idx+1, ref_length):
            if i not in neighbor_ids:
                if len(ref_index) > num_ref:
                    break
                ref_index.append(i)
    return ref_index


# read frame-wise masks
def read_mask(mpath):
    mnames = os.listdir(mpath)
    masks = [None for i in range(len(mnames))]
    fext = mnames[0][-4:]
    mlst = [mpath+'/'+str(name)+fext for name in range(len(mnames))]
    for i in range(len(mnames)):
        m = Image.open(mlst[i])
        m = m.resize((w, h), Image.NEAREST)
        m = np.array(m.convert('L'))
        m = np.array(m > 0).astype(np.uint8)
        m = cv2.dilate(m, cv2.getStructuringElement(
            cv2.MORPH_CROSS, (3, 3)), iterations=4)
        masks[i] = Image.fromarray(m*255)
    return masks


#  read frames from video
def read_frame_from_videos(video_path):
    vname = video_path
    lst = os.listdir(vname)
    fext = lst[0][-4:]
    frames = [None for i in range(len(lst))]
    fr_lst = [vname+'/'+str(name)+fext for name in range(len(lst))]
    for i in range(len(lst)):
        image = cv2.imread(fr_lst[i])
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        frames[i] = (image.resize((w, h)))
    return frames


def set_up_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = importlib.import_module('E2FGVI.model.e2fgvi')
    model = net.InpaintGenerator().to(device)
    ckpt_path = 'E2FGVI/release_model/E2FGVI-CVPR22.pth'
    data = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(data)
    print(f'Loading model from: {ckpt_path}')
    model.eval()
    print('Model setup completed')

    return model, device


def get_images_and_masks(video_path, mask_path):
    print(f'Loading videos and masks from: {video_path}')
    frames = read_frame_from_videos(video_path)
    masks = read_mask(mask_path)
    return frames, masks


def gen_frames_and_masks(final_images, final_masks):
    frames = final_images
    masks = final_masks

    num_of_splits = len(frames)
    num_of_frames = len(frames[0])

    for i in range(num_of_splits):

        imgs = [None for k in range(num_of_frames)]
        for j in range(num_of_frames):
            imgs[j] = (Image.fromarray(cv2.cvtColor(frames[i][j], cv2.COLOR_BGR2RGB))).resize((432, 240))

        ms = [None for k in range(num_of_frames)]
        for j in range(num_of_frames):
            m = Image.fromarray(masks[i][j])
            m = m.resize((432, 240), Image.NEAREST)
            m = np.array(m.convert('L'))
            m = np.array(m > 0).astype(np.uint8)
            m = cv2.dilate(m, cv2.getStructuringElement(
                cv2.MORPH_CROSS, (3, 3)), iterations=4)
            ms[j] = Image.fromarray(m*255)

        frames[i] = imgs
        masks[i] = ms

    return frames, masks

def preprocess_images_and_masks(frames, masks, device):
    imgs = to_tensors()(frames).unsqueeze(0) * 2 - 1
    frames = [np.array(f).astype(np.uint8) for f in frames]
    binary_masks = [np.expand_dims((np.array(m) != 0).astype(np.uint8), 2)
                for m in masks]
    masks = to_tensors()(masks).unsqueeze(0)
    imgs, masks = imgs.to(device), masks.to(device)

    return frames, binary_masks, imgs, masks

def inpaint(frames, binary_masks, imgs, masks, video_length, model):

    comp_frames = [[None for j in range(30)] for i in range(video_length//30 + 1)]

    # completing holes by e2fgvi
    ind = 0
    for start in range(0, video_length, 30):
      vid_len = min(start+30, video_length)
      for f in tqdm(range(start, vid_len, neighbor_stride)):
          neighbor_ids = [i for i in range(max(start, f-neighbor_stride), min(vid_len, f+neighbor_stride+1))]
          ref_ids = get_ref_index(f, neighbor_ids, start, vid_len)
          selected_imgs = imgs[:1, neighbor_ids+ref_ids, :, :, :]
          selected_masks = masks[:1, neighbor_ids+ref_ids, :, :, :]
      
          with torch.no_grad():
              masked_imgs = selected_imgs*(1-selected_masks)
              pred_img, _ = model(masked_imgs, len(neighbor_ids))

              pred_img = (pred_img + 1) / 2
              pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
              for i in range(len(neighbor_ids)):
                  idx = neighbor_ids[i]
                  img = np.array(pred_img[i]).astype(
                      np.uint8)*binary_masks[idx] + frames[idx] * (1-binary_masks[idx])
                  if comp_frames[ind][idx % 30] is None:
                      comp_frames[ind][idx % 30] = img
                  else:
                      comp_frames[ind][idx % 30] = comp_frames[ind][idx % 30].astype(
                          np.float32)*0.5 + img.astype(np.float32)*0.5
      ind += 1
    return comp_frames


def inpaint_main(frames, masks):
    num_of_splits = len(frames)
    num_of_frames = len(frames[0])
    comp_frames = [None for i in range(num_of_splits)]

    model, device = set_up_model()
    
    for i in range(num_of_splits):

      f, binary_masks, imgs, m = preprocess_images_and_masks(frames[i], masks[i], device)
      comp_frames[i] = inpaint(f, binary_masks, imgs, m, num_of_frames, model)
      print('Completed Inpainting', i)

    return comp_frames

def merge(video_length, num_of_splits, new_coords, comp_frames, images):

    ind = -1
    inpainted_frames = [None for i in range(video_length)]
    for i in range(video_length):

        if i%30 == 0:
            ind += 1

        for j in range(num_of_splits):
            inpainted = comp_frames[j][ind][i%30]
            inpainted = cv2.cvtColor(inpainted, cv2.COLOR_RGB2BGR)

            # h = new_coords[j][0][1] - new_coords[j][0][0]
            w = new_coords[j][1][1] - new_coords[j][1][0]

            if(w == 432):
                images[i][new_coords[j][0][0] : new_coords[j][0][1], new_coords[j][1][0] : new_coords[j][1][1]] = inpainted

            if(w < 432):
                images[i][new_coords[j][0][0] : new_coords[j][0][1], new_coords[j][1][0] : new_coords[j][1][1]] = inpainted[:, 0 : w]

        inpainted_frames[i] = images[i]

    return inpainted_frames
