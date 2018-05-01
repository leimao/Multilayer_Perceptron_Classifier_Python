import numpy as np
import matplotlib.pyplot as plt


def plot_curve(losses, accuracies, savefig = True, showfig = False, filename = 'training_curve.png'):

    x = np.arange(len(losses))
    y1 = accuracies
    y2 = losses

    fig, ax1 = plt.subplots(figsize = (12,8))
    ax2 = ax1.twinx()

    ax1.plot(x, y1, color = 'b', marker = 'o', label = 'Training Accuracy')
    ax2.plot(x, y2, color = 'r', marker = 'o', label = 'Training Loss')

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax2.set_ylabel('Loss')

    ax1.legend()
    ax2.legend()

    if savefig:
        fig.savefig(filename, format = 'png', dpi = 600, bbox_inches = 'tight')
    if showfig:
        plt.show()
    plt.close()

    return 


