import matplotlib.pyplot as plt
import numpy as np

def show_image(img, title='', cmap='gray'):
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

def overlay_mask(img, mask, alpha=0.5, cmap='jet'):
    plt.imshow(img, cmap='gray')
    plt.imshow(mask, alpha=alpha, cmap=cmap)
    plt.axis('off')
    plt.show()

def plot_training_curves(logs, metrics=['train_loss', 'val_loss']):
    for m in metrics:
        plt.plot(logs[m], label=m)
    plt.legend()
    plt.grid(True)
    plt.show()