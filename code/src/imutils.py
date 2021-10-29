from PIL import Image, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
from os import path, mkdir
from shutil import rmtree
import av 

TESTIM = 'data/videos/test.jpg'
FRAME = './data/videos/10-22-2021/frames/frame-0000.jpg'
IMAGE = FRAME 

class ImageProcessor:
    def __init__(self, image):
        self.raw_im = image if isinstance(image, Image.Image) else Image.open(image)
        color = ImageEnhance.Color(self.raw_im)
        bw_im = color.enhance(0)
        contrast = ImageEnhance.Contrast(bw_im)
        self.processed_im = contrast.enhance(3.0)
        self.extracted_im_data = []
        self.object_pixels = []
        self.extract_objects()
        
    def extract_objects(self, thresh=0):
        self.extracted_im_data = []
        pix = self.processed_im.load()
        width, height = self.processed_im.size

        for y in range(0, height): 
            col = []
            for x in range(0, width): 
                current_pix = pix[x, y]
                if current_pix[0] <= thresh:
                    col.append((255, 255, 255))
                    self.object_pixels.append((x, y))
                else: 
                    col.append((0, 0, 0))

            self.extracted_im_data.append(col)
        self.extracted_im_data = np.array(self.extracted_im_data)
        return self.extracted_im_data

    def show_pipeline_plot(self):
        f, axarray = plt.subplots(1, 3)

        axarray[0].imshow(self.raw_im)
        axarray[0].title.set_text('Raw image')
        axarray[1].imshow(self.processed_im)
        axarray[1].title.set_text('Processed image')
        axarray[2].imshow(self.extracted_im_data) 
        axarray[2].title.set_text('Extracted features')
        plt.show()

    def show_extracted_objects(self):
        plt.imshow(self.extracted_im_data)
        plt.show()

    def save_extracted_objects(self, path):
        size=(len(self.extracted_im_data[0]), 
                len(self.extracted_im_data))
        Image.fromarray(np.uint8(self.extracted_im_data)).save(path)
        

    def get_object_pixels(self):
        return self.object_pixels


def get_centroid(coordinates): 
    x, y = np.transpose(coordinates)
    x_mean = np.sum(x) / len(x)
    y_mean = np.sum(y) / len(y)
    return x_mean, y_mean 


def get_frame_centroids(file):
    container = av.open(file)
    centroids = []
    if path.isdir(path.join(path.dirname(file), 'frames')):
        rmtree(path.join(path.dirname(file), 'frames'))
    mkdir(path.join(path.dirname(file), 'frames'))
    mkdir(path.join(path.dirname(file), 'frames/raw'))
    mkdir(path.join(path.dirname(file), 'frames/extracted'))


    frames = container.decode(video=0)
    n_frames = container.streams.video[0].frames
    i = 0
    for frame in container.decode(video=0):
        im = frame.to_image()
        im.save(path.join(path.dirname(file), 
            'frames/raw/frame-%04d.jpg' % frame.index))
        ip = ImageProcessor(im)
        centroids.append(get_centroid(ip.get_object_pixels()))
        ip.save_extracted_objects(path.join(path.dirname(file),
            'frames/extracted/frame-%04d-extracted.jpg' % frame.index))
        i += 1

        # temporary, frames past 100 have artifacts
        if i > 100:
            break
        if i % (n_frames / 10) == 0:
            print("Processing frames: ", np.round(i / n_frames * 100, 2), '%')
        
    # use framerate to obtain t 
    fr = 29.97
    t = np.arange(0, fr * len(centroids), fr)
    return np.append(np.transpose(centroids), [t], axis=0)
    

# im = ImageProcessor(Image.open(FRAME))
# im.save_extracted_objects('./data/test.jpg')

file = './data/videos/10-22-2021/test.MOV'
x, y, t = get_frame_centroids(file)
zeros = np.zeros(len(x))

np.savetxt(path.join(path.dirname(file), 'data.csv'),
            np.append([x, y, t], [zeros, zeros], axis=0),
            delimiter=',',
            header='x,y,t,dphi,state')