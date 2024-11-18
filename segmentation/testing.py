import json
import cv2 as cv
import os
import numpy as np


folder_path = '/home/muhammad-ali/working/hybrid_detector/base_data/images'
save = '/home/muhammad-ali/working/hybrid_detector/base_data/processed'

circles = json.load(open('/home/muhammad-ali/Downloads/all.json'))
i = 0
for img_name, circles in circles.items():
    if len(circles)> 0:
        img_path = os.path.join(folder_path, img_name)
        img = cv.imread(img_path)
        # Calculate a scaling factor based on image size
        scale_factor = min(img.shape[1], img.shape[0]) / 500  # 500 is an arbitrary base size
        font_scale = max(0.5 * scale_factor, 0.3)//2  # Avoid too small font
        thickness = max(int(2 * scale_factor), 1)//2  # Ensure thickness is at least 1

        # Find min, max radius and count of circles
        radii = [circle[2] for circle in circles]
        max_radius = max(radii)
        min_radius = min(radii)
        circle_count = len(circles)

        # Draw each circle and annotate

        for x, y, r in circles:
            # Draw the circle
            x,y,r = int(x), int(y), int(r)
            cv.circle(img, (x, y), r, (0, 255, 0), thickness)
            # Write the coordinates and radius at the center of each circle
            cv.putText(img, f"{r}", (x, y), cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

        # Write max, min radius and circle count at the bottom left corner
        bottom_left_text = f"Max Radius: {max_radius}, Min Radius: {min_radius}, Count: {circle_count}"
        cv.putText(img, bottom_left_text, (10, img.shape[0] - 10), cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        # Save or display the annotated image
        output_path = os.path.join(save, img_name)
        cv.imwrite(output_path, img)
        i+= 1
        if i%100 ==0:
            print(f'{i} images done')
