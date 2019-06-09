import numpy as np
import scipy.misc
from scipy.misc import imsave
def squarify_predictions(predictions):
    '''
        Print the predictions by target classifier, with each class containing 10 images in the same row. 
    '''
    for i in range(10):
        preds = " ".join(map(str, predictions[i*10: (i+1)*10]))
    return preds

def save_images(X, save_path):
  # [-1, 1] -> [0,255]
  if isinstance(X.flatten()[0], np.floating):
    X = ((X + 1.) * 127.5).astype('uint8')

  n_samples = X.shape[0]
  rows = int(np.sqrt(n_samples))
  while n_samples % rows != 0:
    rows -= 1

  nh, nw = rows, n_samples // rows

  if X.ndim == 2:
    X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

  if X.ndim == 4:
    h, w = X[0].shape[:2]
    img = np.zeros((h * nh, w * nw, 3))
  elif X.ndim == 3:
    h, w = X[0].shape[:2]
    img = np.zeros((h * nh, w * nw))

  for n, x in enumerate(X):
    j = n // nw
    i = n % nw
    img[j * h:j * h + h, i * w:i * w + w] = x

  imsave(save_path, img)