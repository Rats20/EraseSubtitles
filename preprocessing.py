# Importing Libraries
import os
import cv2
import numpy as np
import moviepy.editor as mp


# To generate image frames and audio from the video
def gen_image_frames(name, ip_path, a_path):
 
    if not os.path.exists(a_path):
        os.makedirs(a_path)

    vidcap = cv2.VideoCapture(ip_path + name)

    imgs = []
    count = 0
    success,image = vidcap.read()
    while success:
        imgs.append(image)
        success,image = vidcap.read()
        count += 1

    my_clip = mp.VideoFileClip(ip_path + name)
    my_clip.audio.write_audiofile(a_path + name[:-4] + '.mp3')

    return imgs


# Color segmentation
def seg(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 200])
    upper = np.array([150, 15, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return mask

def seg_imgs(images):
    masks = [None for i in range(len(images))]
    for i in range(len(images)):
        masks[i] = seg(images[i])
    return masks