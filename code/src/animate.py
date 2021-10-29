import av 
from os import path, mkdir, listdir
from shutil import rmtree
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image
import cv2


def make_centroid_animation(frames, centroid_data, output_dir, filename, fps=29.97): 
    # Pick shortest array to avoid index error
    num_ani_frames = len(frames) if len(frames) < len(centroid_data) else len(centroid_data) 

    tmp_frames_dir = path.join(output_dir, f"{path.splitext(filename)[0]}-tmp-frames")
    if path.isdir(tmp_frames_dir):
        rmtree(tmp_frames_dir)
    mkdir(tmp_frames_dir)

    for i in range(0, num_ani_frames):
        plt.imshow(frames[i], origin='lower') 
        x, y = centroid_data[i]
        plt.scatter(x, y)
        plt.savefig(path.join(tmp_frames_dir,
            'frame-%04d-animation.jpg' % i))
        plt.clf()
    
    ani_images = sorted([img for img in listdir(tmp_frames_dir) if img.endswith('.jpg')])
    # sorted in reverse for some reasom 
    ani_images.reverse()

    # https://stackoverflow.com/questions/44947505/how-to-make-a-movie-out-of-images-in-python?answertab=oldest#tab-top
    ani_frame = cv2.imread(path.join(tmp_frames_dir, ani_images[0]))
    height, width, layers = ani_frame.shape

    video = cv2.VideoWriter(path.join(output_dir, filename), 0, fps, (width, height))

    for ani_image in ani_images:
        video.write(cv2.imread(path.join(tmp_frames_dir, ani_image)))

    cv2.destroyAllWindows()
    video.release()
    rmtree(tmp_frames_dir)



input_dir = './data/videos/10-22-2021/frames/extracted/'
ims = sorted([path.join(input_dir, img) for img in listdir(input_dir) if img.endswith('.jpg')])
ims = [Image.open(img) for img in ims]
x, y, t, dphi, state = np.loadtxt('./data/videos/10-22-2021/data.csv', delimiter=',')

make_centroid_animation(ims, np.transpose([x, y]), './animations/exp/', 'test_ani.avi')