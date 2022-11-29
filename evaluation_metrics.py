import os
import cv2
import numpy as np
import math

out_dir='/content/drive/MyDrive/O/'
gt_dir='/content/drive/MyDrive/VideoDataset/subVideos/Original/'

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()



psnr_list=[]
ssim_list=[]
for i in x:
  video1=cv2.VideoCapture(gt_dir+i)
  video2=cv2.VideoCapture(out_dir+i)
  success1,image1=video1.read()
  success2,image2=video1.read()
  count=0
  psnr_val=0
  ssim_val=0
  while success2 and success1:
    psnr_val+=psnr(image1,image2)
    ssim_val+=ssim(image1,image2)
    success1,image1=video1.read()
    success2,image2=video1.read()
    count += 1
  if count!=0:
    psnr_list.append(psnr_val/count)
    ssim_list.append(ssim_val/count)
print("PSNR is: ",sum(psnr_list)/len(psnr_list))
print("SSIM is: ",sum(ssim_list)/len(ssim_list))

