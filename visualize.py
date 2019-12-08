import numpy as np
import argparse
import os
import imageio

parser = argparse.ArgumentParser(description='image_writer')
parser.add_argument('--file', type=str, default='img', help='npy file to save to png')
args = parser.parse_args()

data = np.load(args.file)
_, channels, width, height = data.shape
if channels == 1:
    data = data.reshape((width, height, channels))
elif channels == 3:
    data = np.transpose(data, (0, 2, 3, 1)).reshape((width, height, channels))

filename, file_extension = os.path.splitext(args.file)
imageio.imwrite(filename + ".jpg", data)

