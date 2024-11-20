from PIL import Image, ImageDraw
import os
import numpy as np
from dataloader import orig_train_batches

def visualizer(save: str):
    z = 0
    for batch in orig_train_batches:
        imgs, targets = batch
        for i in range(len(imgs)):
            # Convert numpy array to PIL Image
            img_array = (imgs[i].numpy() + 1) * 127.5
            img = Image.fromarray(img_array.astype('uint8'))
            draw = ImageDraw.Draw(img)
            
            target = targets[i].numpy()
            for circle in target:
                circle = np.array(circle)
                if np.any(circle != [0,0]):
                    x, y = int(circle[0]), int(circle[1])
                    r = 2
                    draw.ellipse([x-r, y-r, x+r, y+r], fill=(0, 255, 0))
            
            z += 1
            output_path = os.path.join(save, f'{z}.jpg')
            img.save(output_path)
            
            if z % 100 == 0:
                print(f'{z} images done')
    return

if __name__ == '__main__':
    # it assumes the size of the image corresponds to the target
    save_path = '/mloscratch/sayfiddi/pointer/base_data/processed'
    visualizer(save_path)