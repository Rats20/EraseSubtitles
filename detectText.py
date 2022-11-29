from text_detection.predict import get_text_boxes
import cv2

def resize_img(image):
    h, w = image.shape[:2]
    rescale_fac = max(h, w) / 1000
    if rescale_fac > 1.0:
        h = int(h / rescale_fac)
        w = int(w / rescale_fac)
    return h, w, rescale_fac

def get_coords(num_of_frames, masks):
    xmin, ymin, xmax, ymax = 10000, 10000, 0, 0
    new_coords = [xmin, ymin, xmax, ymax]
    rh, rw, rescale_fac = resize_img(masks[0])
    max_height, max_width = masks[0].shape[:2]
    print('Original Dimensions: ', max_height, 'x', max_width)
    print('Rescaled Dimensions: ', rh, 'x', rw)

    for i in range(num_of_frames):
        input_img = cv2.resize(masks[i], (rw,rh))
        text = get_text_boxes(input_img)

        for coord in text:
            x = coord[::2]
            y = coord[1::2]
            xmin = min(x)
            ymin = min(y)
            xmax = max(x)
            ymax = max(y)
        
        if new_coords[0] > xmin:
            new_coords[0] = xmin
        
        if new_coords[1] > ymin:
            new_coords[1] = ymin

        if new_coords[2] < xmax:
            new_coords[2] = xmax
        
        if new_coords[3] < ymax:
            new_coords[3] = ymax

    new_coords = [int(coord) for coord in new_coords]
    # print('New_coords: ', new_coords)

    xmin, ymin, xmax, ymax = new_coords

    # Inverse rescaling (getting back to original coordinates)
    xmin = int(xmin * rescale_fac)
    if xmin - 10 >= 0:
      xmin -= 10
      
    ymin = int(ymin * rescale_fac)
    if ymin - 10 >= 0:
      ymin -= 10

    xmax = int(xmax * rescale_fac)
    if xmax + 10 <= max_width:
      xmax += 10

    ymax = int(ymax * rescale_fac)
    if ymax + (20 * rescale_fac) <= max_height:
      ymax += int(20 * rescale_fac)

    return [xmin, ymin, xmax, ymax]
