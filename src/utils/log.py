import numpy as np


def constraints_image(sample, constraints):

    image = np.zeros((1, sample.shape[1], sample.shape[2], 3))
    if sample.shape[-1] == 1:
        image += np.stack((np.squeeze((sample+1)/2, axis=-1),) * 3, -1)
        image = (image * 255)

    else:
        image += (sample * 255)

    for x in range(image.shape[1]):
        for y in range(image.shape[2]):
            if np.sum(constraints[0, x, y]) != 0:
                pix_sqerr = np.square(
                    np.sum(sample[0, x, y] - constraints[0, x, y]))
                if pix_sqerr > 0.1*sample.shape[-1]:
                    image[0, x, y] = [255, 0, 0]
                else:
                    image[0, x, y] = [0, 255, 0]

    return image
