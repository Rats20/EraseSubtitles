import cv2
import numpy as np

def split(height, width, max_height, max_width, coordinates):
    # ymax - ymin
    h = coordinates[3] - coordinates[1]
    # xmax - xmin
    w = coordinates[2] - coordinates[0]

    # Number of slice cut wrt ht n width (imagine, like a grid)
    # Assumption: subtitles ht < 240
    # Check if ymax is in upper half or lower half

    if h > 240:
      # This could be due to the inclusion of remote scene text like watermarks
      # Subtitles could be at top or bottom
      ht = [ coordinates[1] + height, coordinates[1], coordinates[3], coordinates[3] - height]

    else:

      # Top subtitles
      if coordinates[3] < max_height//2 :
        # ht and width coords to construct the grid
        # Top-left coordinate (ymin)
        ht = [coordinates[1] + height, coordinates[1]]

      # Bottom subtitles
      else:
        # Bottom-left coordinate (ymax)
        ht = [coordinates[3], coordinates[3] - height]

    w_count = w//width

    wt = []
    # Top-left coordinate (xmin)
    wt.append(coordinates[0])

    ind = 0
    while(w_count >= 0):
      wt.append(wt[ind] + width)
      w_count -= 1
      ind += 1

    # Checking the last coordinate of width
    #(because it is possible that the last coord could be out of bounds)
    # if out of bounds, then the image requires padding in the next step
    if wt[-1] > max_width:
      wt[-1] = max_width

    new_coords = []

    if h > 240:
      # All top splits
      for j in range(len(wt)-1):
        new_coords.append([ [ht[1], ht[0]] , [wt[j],wt[j+1]] ])

      # All bottom splits
      for j in range(len(wt)-1):  
        new_coords.append([ [ht[3], ht[2]] , [wt[j],wt[j+1]] ])

      num_of_splits = (len(wt)-1)*2

      return new_coords, num_of_splits

    else:
      for j in range(len(wt)-1):
        new_coords.append([ [ht[1], ht[0]] , [wt[j],wt[j+1]] ])

    num_of_splits = len(wt)-1

    return new_coords, num_of_splits


def gen_regions(h, w, images, masks, coordinates):

    max_height, max_width = images[0].shape[:2]

    new_coords, num_of_splits = split(h, w, max_height, max_width, coordinates)

    num_of_frames = len(images)

    # Storing the (subtitle text) frame regions that have to be inpainted
    j = 0
    imgs = [[None for j in range(num_of_frames)]for i in range(num_of_splits)]
    final_masks = [[None for j in range(num_of_frames)]for i in range(num_of_splits)]

    for i in range(num_of_frames):

      im = images[i]
      mask = masks[i]
      # mask = cv2.dilate(mask, np.ones((3,3),np.uint8), iterations = 2)

      for j in range(num_of_splits):
        # h = new_coords[j][0][1] - new_coords[j][0][0]
        w = new_coords[j][1][1] - new_coords[j][1][0]

        if(w == 432):
          imgs[j][i] = im[new_coords[j][0][0] : new_coords[j][0][1], new_coords[j][1][0] : new_coords[j][1][1]]
          final_masks[j][i] = mask[new_coords[j][0][0] : new_coords[j][0][1], new_coords[j][1][0] : new_coords[j][1][1]]
        
        #Padding
        elif(w < 432):
          imgs[j][i] = im[new_coords[j][0][0] : new_coords[j][0][1], new_coords[j][1][0] : new_coords[j][1][1]]
          final_masks[j][i] = mask[new_coords[j][0][0] : new_coords[j][0][1], new_coords[j][1][0] : new_coords[j][1][1]]
          extra = 432 - w
          # [t, b, l, r]
          imgs[j][i] = cv2.copyMakeBorder(imgs[j][i], 0, 0, 0, extra, borderType=cv2.BORDER_CONSTANT, value=[0,0,0])
          final_masks[j][i] = cv2.copyMakeBorder(final_masks[j][i], 0, 0, 0, extra, borderType=cv2.BORDER_CONSTANT, value=[0,0,0])
      
    return new_coords, num_of_splits, imgs, final_masks
