import numpy as np

def normalize_images(images):
    return images / 255.0

def pad_to_square(images, target_size=32):
    if len(images.shape) == 2:
        height, width = images.shape
        padded = np.zeros((target_size, target_size))
        h_offset = (target_size - height) // 2
        w_offset = (target_size - width) // 2
        padded[h_offset:h_offset+height, w_offset:w_offset+width] = images
        return padded
    else:
        batch_size, height, width = images.shape
        padded = np.zeros((batch_size, target_size, target_size))
        h_offset = (target_size - height) // 2
        w_offset = (target_size - width) // 2
        padded[:, h_offset:h_offset+height, w_offset:w_offset+width] = images
        return padded
