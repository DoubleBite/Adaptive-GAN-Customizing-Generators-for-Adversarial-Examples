import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import itertools
import numpy as np


def plot_images(images, epoch, result_dir=None):
    '''
        Plot 100 images produced by the generator, with each class containing 10 images in the same row. 
    '''
    size_figure_grid = 10
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))

    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)
        ax[i, j].imshow(images[i*size_figure_grid+j], cmap="gray")

    label = 'Epoch {0}'.format(epoch)
    fig.text(0.5, 0.04, label, ha='center')

    if result_dir:
        plt.savefig(os.path.join(result_dir,'{0}.png'.format(epoch)))    
    plt.close()
    return
    
    
    
def plot_losses(d_losses, g_losses):
    '''
        Plot the loss curve for generator and discriminator 
    '''
    plt.title('Losses', fontsize=20)
    plt.plot(np.array(d_losses), label="d_loss")
    plt.plot(np.array(g_losses), label="g_loss")
    plt.legend()
    plt.show()    
    return 