import os
import cv2
import argparse

import moviepy
from moviepy.editor import VideoFileClip, AudioFileClip

from preprocessing import gen_image_frames, seg_imgs
from detectText import get_coords
from splitRegion import gen_regions
from inpaint import gen_frames_and_masks, inpaint_main, merge

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="eraseSubtitles")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--savefps", type=int, default=30)
    args = parser.parse_args()

    print('Starting...')
    print('\nPreprocessing...')

    name = args.video
    frame_rate = args.savefps

    ip_path = 'Input/Video/'
    a_path = 'Input/Audio/'

    images = gen_image_frames(name, ip_path, a_path)
    masks = seg_imgs(images)
    num_of_frames = len(masks)

    print('Stats of the video')
    print('Number of frame: ', num_of_frames)


    print('\nDetecting Text...')
    max_height, max_width = images[0].shape[:2]
    coords = get_coords(num_of_frames, masks)
    print('Subtitle Region coords: ', coords)
    print('Subtitle Text Detection Done')

    if(coords != []):

        h, w = 240, 432
        print('\nSplitting subtitle region to', h, 'x', w, 'parts...')
        new_coords, num_of_splits, final_images, final_masks = gen_regions(h, w, images, masks, coords)
        print('Split coords: ', new_coords)
        print('Splitting Concluded')

        print('\nInpainting Begins...')
        iframes, imasks = gen_frames_and_masks(final_images, final_masks)
        comp_frames = inpaint_main(iframes, imasks)
        inpainted_frames = merge(num_of_frames, num_of_splits, new_coords, comp_frames, images)
        print('Inpainting The End')


        im = inpainted_frames[0]
        h, w = im.shape[:2]
        size = (w, h)

        save_path = 'Output/Inpainted/' + name
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"MP4V"), frame_rate, size)

        print('\nStoring video in ', save_path)

        for i in range(num_of_frames):
            out.write(inpainted_frames[i])
        out.release()

        print('\nAdding Audio')

        video_clip = VideoFileClip('Output/Inpainted/' + name[:-4] + '.mp4')
        audio_clip = AudioFileClip('Input/Audio/'+ name[:-4] +'.mp3')

        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile('Output/' + name[:-4] + '.mp4')

    else:
      print('No Subtitles found in the input video!!!')

    print('\nCompleted :)')
