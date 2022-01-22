from os import path, mkdir, listdir
from shutil import rmtree
import matplotlib.pyplot as plt 
from matplotlib.collections import LineCollection
import cv2
import sys 
import numpy as np
from .mathutils import pos_angle2vertices



def make_centroid_animation(frames, centroid_data, output_dir, filename, fps=29.97): 
    # Pick shortest array to avoid index error
    num_ani_frames = len(frames) if len(frames) < len(centroid_data) else len(centroid_data) 

    tmp_frames_dir = path.join(output_dir, f"{path.litext(filename)[0]}-tmp-frames")
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
        if i % int(num_ani_frames / 10) == 0:
            sys.stdout.write(f"\rProcessing frames: {np.round(i / num_ani_frames * 100, 2)}%")
        
    sys.stdout.write("\n")
    frames2vid(tmp_frames_dir, output_dir, filename, fps=fps)


def frames2vid(tmp_frames_dir, output_dir, filename, file_ext='.jpg', fps=29.97):
    ani_images = sorted([img for img in listdir(tmp_frames_dir) if img.endswith(file_ext)])
    # sorted in reverse for some reason
    # ani_images.reverse()

    # https://stackoverflow.com/questions/44947505/how-to-make-a-movie-out-of-images-in-python?answertab=oldest#tab-top
    ani_frame = cv2.imread(path.join(tmp_frames_dir, ani_images[0]))
    height, width, layers = ani_frame.shape

    video = cv2.VideoWriter(path.join(output_dir, filename), 0, fps, (width, height))

    for ani_image in ani_images:
        video.write(cv2.imread(path.join(tmp_frames_dir, ani_image)))

    cv2.destroyAllWindows()
    video.release()
    rmtree(tmp_frames_dir)


def make_swimmer_animation(swimmer_data, t, output_dir, filename, fps=29.97, dpi=150, xlim=(-10, 10), ylim=(-10, 10)):
    """
    
    Params
    ------
    in format ([s1_positions, s1_phis, s1_h, s1_w], ...)
    """
    tmp_frames_dir = path.join(output_dir, f"{path.splitext(filename)[0]}-tmp-frames")
    if path.isdir(tmp_frames_dir):
        rmtree(tmp_frames_dir)
    mkdir(tmp_frames_dir)
    ax = plt.axes()
    
    
    for i in range(0, len(t)): 
        ax.set_aspect('equal')
        ax.set_ylim(*xlim)
        ax.set_xlim(*ylim)
        lcs = []
        for s in swimmer_data.keys():
            x, y = swimmer_data[s]['pos'][i]
            phi = swimmer_data[s]['phi'][i]
            verts = pos_angle2vertices(x, y, phi, swimmer_data[s]['h'], swimmer_data[s]['w'])
            verts.append(verts[0])
            verts = np.array(verts)
            lcs.append(LineCollection([np.column_stack(np.transpose(verts))]))

        for l in lcs: 
            ax.add_collection(l)
        ax.text(0.05, 0.05, "t=" + str(round(t[i], 3)), horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)
        plt.savefig(path.join(tmp_frames_dir, 'frame-%04d-animation.jpg' % i), dpi=dpi)
        ax.cla()
    frames2vid(tmp_frames_dir, output_dir, filename, fps=fps)