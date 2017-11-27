import h5py
import os
import sys
import numpy as np
from PIL import Image


def sample_to_png(samples_path, images_path):
    for sample in os.listdir(samples_path + "/"):
        f = h5py.File(samples_path + "/" + sample)
        data = np.asarray(f["features"])
        f.flush()
        f.close()
        out = np.squeeze(data + 1.0) * 128
        image = Image.fromarray(out.astype("uint32"))
        image.convert("L").save(images_path + "/" + sample + ".png")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        sample_to_png(sys.argv[1], sys.argv[2])
    else:
        sample_to_png("samples/", "images/")
