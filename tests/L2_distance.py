import numpy as np
from PIL import Image


if __name__ == "__main__":
    # Open an image file
    img1 = Image.open("E:/mingchao_data/images_for_eval/speed_up.png")
    img2 = Image.open("E:/mingchao_data/images_for_eval/forward.png")
    # Assuming img1 and img2 are numpy arrays representing the images
    # and they have the same dimensions
    img1 = np.array(img1)  # Convert to numpy array if not already
    img2 = np.array(img2)

    # Flatten the images
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()

    # Calculate L2 distance
    l2_distance = np.sqrt(np.sum((img1_flat - img2_flat) ** 2))

    print("L2 Distance:", l2_distance)
