import os
import cv2

import moviepy
from moviepy.editor import VideoFileClip, AudioFileClip

from preprocessing import gen_image_frames, seg_imgs
from detectText import get_coords
from splitRegion import gen_regions
from inpaint import gen_frames_and_masks, inpaint_main, merge

if __name__ == '__main__':

    print('Starting...')
    print('\nPreprocessing...')

    name = '43.mp4'

    ip_path = 'Input/Video/'
    a_path = 'Input/Audio/'

    images = gen_image_frames(name, ip_path, a_path)
    masks = seg_imgs(images)
    num_of_frames = len(masks)

    print('Stats of the video')
    print('Number of frame: ', num_of_frames)


    print('\nDetecting Text...')
    
    coords = get_coords(num_of_frames, masks)
    print('Subtitle Region coords: ', coords)
    print('Subtitle Text Detection Done')


    h, w = 240, 432
    print('\nSplitting subtitle region to', h, 'x', w, 'parts...')
    new_coords, num_of_splits, final_images, final_masks = gen_regions(h, w, images, masks, coords)
    print('Split coords: ', new_coords)
    print('Splitting Concluded')

    print('\nInpainting Begins...')
    iframes, imasks = gen_frames_and_masks(final_images, final_masks)
    comp_frames = inpaint_main(iframes, imasks)
    inpainted_frames = merge(num_of_frames, num_of_splits, new_coords, comp_frames, images)
    print('\nInpainting The End')


    im = inpainted_frames[0]
    h, w = im.shape[:2]
    size = (w, h)

    save_path = 'Output/Inpainted/' + name
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"MP4V"), 30, size)

    print('Storing video in ', save_path)

    for i in range(num_of_frames):
        out.write(inpainted_frames[i])
    out.release()

    video_clip = VideoFileClip('Output/Inpainted/' + name[:-4] + '.mp4')
    audio_clip = AudioFileClip('Input/Audio/'+ name[:-4] +'.mp3')

    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile('Output/Final/' + name[:-4] + '_final.mp4')

    print('\nCompleted :)')