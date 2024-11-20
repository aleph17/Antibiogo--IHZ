import os
import numpy as np
import cv2 as cv

def padding(img, desired_size):
    """generates an image with padded with (0,0,0) to the desired_size"""
    old_image_height, old_image_width, channels = img.shape
    new_image_height, new_image_width = desired_size
    # create new image of desired size and color (black) for padding
    color = (0,0,0)
    result = np.full((new_image_height,new_image_width, channels), color, dtype=np.uint8)

    # compute center offset
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2

    # copy img image into center of result image
    result[y_center:y_center+old_image_height, x_center:x_center+old_image_width] = img

    return result

archive = '/home/muhammad-ali/working/hungarian/base_data'

img_dir = os.path.join(archive, 'images')
padded_im = '/home/muhammad-ali/working/hungarian/base_data/ready'

img_list = sorted(list(os.listdir(img_dir)))

img_dirs = [os.path.join(img_dir, x) for x in img_list]

for i in range(len(img_dirs)):
    image = cv.imread(img_dirs[i])
    # mask = PIL.Image.open(mask_dirs[i])
    maxim = max(image.shape)
    image= padding(image, (maxim, maxim))
    image = cv.resize(image, (1024, 1024), interpolation=cv.INTER_LINEAR)
    cv.imwrite(str(padded_im + f"/{img_list[i]}"), image)
    if i%100 == 0:
        print(f'{i} done')
    # mask.save(str(padded_ms + f"/{mask_list[i]}"), 'JPEG')

