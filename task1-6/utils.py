import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_img_hist(images, title: str = "Image and Histogram", labels:list = None, block=True) -> None:

    # if (not isinstance(images, list)) or (not isinstance(images,np.array)):
    #     raise ValueError("Input images as a list or just one image, e.g. [image] or image")
    if not images:
        raise ValueError("images must not be empty")
    if labels is not None and len(labels) != len(images):
        raise ValueError("labels must have the same length as images")

    if type(images) is np.array:
        num_img = 1
    elif type(images) is list:
        num_img = len(images)
    else: 
        raise ValueError("Input images as a list or just one image, e.g. [image] or image")

    _, axes = plt.subplots(num_img, 2, figsize=(15, 3 * num_img), squeeze=False)

    for i, img in enumerate (images):
        # Display the image
        axes[i,0].imshow(img, cmap='gray')
        axes[i,0].axis('off')
        if labels is None:
            axes[i,0].set_title("Image")
        else:
            axes[i,0].set_title(f"{labels[i]}")

        # Display the histogram
        axes[i,1].hist(img.ravel(), bins=256, range=(0, 256), color="black")
        axes[i,1].set_xlabel("Pixel intensity")
        axes[i,1].set_ylabel("Pixel count")
        if i==0:
            axes[i,1].set_title("Intensity Histogram")
        
    plt.suptitle(title)
    plt.show(block=block)

