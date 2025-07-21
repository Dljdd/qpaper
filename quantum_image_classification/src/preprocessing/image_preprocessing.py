import numpy as np

def normalize_images(images):
    """Normalize pixel values to [0, 1] range"""
    return images / 255.0

def pad_to_square(image, target_size=32):
    """Pad image to target size (default 32x32)"""
    if len(image.shape) == 2:
        # Single image
        height, width = image.shape
        padded = np.zeros((target_size, target_size))
        h_offset = (target_size - height) // 2
        w_offset = (target_size - width) // 2
        padded[h_offset:h_offset+height, w_offset:w_offset+width] = image
        return padded
    else:
        # Batch of images
        batch_size, height, width = image.shape
        padded = np.zeros((batch_size, target_size, target_size))
        h_offset = (target_size - height) // 2
        w_offset = (target_size - width) // 2
        padded[:, h_offset:h_offset+height, w_offset:w_offset+width] = image
        return padded

def extract_bitplanes(image, num_bits=8):
    """Extract bitplanes from an image"""
    bitplanes = []
    for bit_pos in range(num_bits):
        # Create a mask for the current bit position
        mask = 1 << bit_pos
        # Extract the bit at position bit_pos for all pixels
        bitplane = ((image * 255).astype(np.uint8) & mask) >> bit_pos
        bitplanes.append(bitplane)
    return bitplanes
