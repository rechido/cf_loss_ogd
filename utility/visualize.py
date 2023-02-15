import numpy as np
import matplotlib.pyplot as plt

def plot_curve_error(data_mean, data_std, x_label, y_label, title, filename=None, show=True, xscale='linear', yscale='linear'):

    fig = plt.figure(figsize=(8, 6))
    plt.title(title)

    alpha = 0.3
    
    plt.plot(range(len(data_mean)), data_mean, '-', color = 'red')
    if data_std is not None:
        plt.fill_between(range(len(data_mean)), data_mean - data_std, data_mean + data_std, facecolor = 'blue', alpha = alpha) 
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.xscale(xscale)
    plt.yscale(yscale)

    plt.tight_layout()
    
    if show:
        plt.show()
    else:
        plt.close(fig)

    if filename is not None:

        fig.savefig(filename)

    plt.figure().clear()
    plt.close('all')
    plt.cla()
    plt.clf()

    pass

def plot_curve_error2(data1_mean, data1_std, data1_label, data2_mean, data2_std, data2_label, x_label, y_label, title, filename=None, show=True):
    
    fig = plt.figure(figsize=(8, 6))
    plt.title(title)

    alpha = 0.3

    plt.plot(range(len(data1_mean)), data1_mean, '-', color = 'blue', label = data1_label)
    if data1_std is not None:
        plt.fill_between(range(len(data1_mean)), data1_mean - data1_std, data1_mean + data1_std, facecolor = 'blue', alpha = alpha)

    plt.plot(range(len(data2_mean)), data2_mean, '-', color = 'red', label = data2_label)
    if data2_std is not None:
        plt.fill_between(range(len(data2_mean)), data2_mean - data2_std, data2_mean + data2_std, facecolor = 'red', alpha = alpha)

    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.close(fig)

    if filename is not None:

        fig.savefig(filename)

    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    pass

def plot_image_grid(data, nRow, nCol, filename=None, show=True):

    size_col = 1.5
    size_row = 1.5

    fig, axes = plt.subplots(nRow, nCol, constrained_layout=True, figsize=(nCol * size_col, nRow * size_row))
    
    data = data.detach().cpu()

    if nRow > 1:
        for i in range(nRow):
            for j in range(nCol):
                k = i * nCol + j
                image = np.squeeze(data[k], axis=0)
                axes[i, j].imshow(image, cmap='gray', vmin=0, vmax=1)
                axes[i, j].xaxis.set_visible(False)
                axes[i, j].yaxis.set_visible(False)

    else:
        for j in range(nCol):
            image   = np.squeeze(data[j], axis=0)
            axes[j].imshow(image, cmap='gray', vmin=0, vmax=1)
            axes[j].xaxis.set_visible(False)
            axes[j].yaxis.set_visible(False)
            
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    if filename is not None:

        fig.savefig(filename)
        
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    pass

def plot_image_grid2(data, nRow, nCol, fig_width=20, fig_height=20, filename=None, show=True):

    fig, axes = plt.subplots(nRow, nCol, constrained_layout=True, figsize=(fig_width, fig_height))
    
    data = data.detach().cpu()

    if nRow > 1 and nCol > 1:
        for i in range(nRow):
            for j in range(nCol):
                k = i * nCol + j
                image = np.squeeze(data[k], axis=0)
                axes[i, j].imshow(image, cmap='gray', vmin=image.min(), vmax=image.max())
                axes[i, j].xaxis.set_visible(False)
                axes[i, j].yaxis.set_visible(False)

    elif nCol > 1:
        for j in range(nCol):
            image   = np.squeeze(data[j], axis=0)
            axes[j].imshow(image, cmap='gray', vmin=image.min(), vmax=image.max())
            axes[j].xaxis.set_visible(False)
            axes[j].yaxis.set_visible(False)

    elif nRow > 1:
        for i in range(nRow):
            image = np.squeeze(data[i], axis=0)
            axes[i].imshow(image, cmap='gray', vmin=image.min(), vmax=image.max())
            axes[i].xaxis.set_visible(False)
            axes[i].yaxis.set_visible(False)
            
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    if filename is not None:

        fig.savefig(filename)
        
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    pass