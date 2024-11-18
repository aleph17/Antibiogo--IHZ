from dataloader import single_batch
import tensorflow as tf
from utils import display,create_mask
from callback import checkpoint_filepath

if __name__=="__main__":
    model = tf.keras.models.load_model(checkpoint_filepath)
    for images, masks in single_batch:
        sample_image, sample_mask = images[0], masks[0]
    
    def show_predictions(dataset=None, num=1):
        if dataset:
            for image, mask in dataset.take(num):
                pred_mask = model.predict(image)
                mask0 = mask[0]
                display([image[0], mask0[...,tf.newaxis], create_mask(pred_mask)])
        else:
            display([sample_image, 
                     sample_mask[...,tf.newaxis],
                     create_mask(model.predict(sample_image[tf.newaxis, ...]))])
    
    #show_predictions(dataset=,num=3)
    pass
