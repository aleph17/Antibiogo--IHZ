import cv2 as cv
import os
import numpy as np
from dataloader import orig_train_batches

def visualizer(save: str):
    z = 0
    for batch in orig_train_batches:
        imgs, targets = batch
        print(imgs.shape)
        for i in range(len(imgs)):
            img, target = imgs[i].numpy(), targets[i].numpy()
            img = (img + 1) * 127.5
            for circle in target:
                circle = np.array(circle)
                if np.any(circle != [0,0,0]):
                    x, y, r = int(circle[0]), int(circle[1]), int(circle[2])
                    cv.circle(img, (x, y), r, (0, 255, 0), 2)
                # Write the coordinates and radius at the center of each circle
                # cv.putText(img, f"({x},{y}), r={r}", (x, y), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            z += 1
            output_path = os.path.join(save, f'{z}.jpg')
            cv.imwrite(output_path, img)
            if z % 100 == 0:
                print(f'{z} images done')
    return
if __name__ == '__main__':
    # it assumes the size of the image corresponds to the target
    save_path = '/hungarian/base_data/processed'
    visualizer(save_path)