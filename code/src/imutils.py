from PIL import Image, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt

PIX_THRESH = 0

im = Image.open("./data/videos/test.jpg")
color = ImageEnhance.Color(im)
bw_im = color.enhance(0)
contrast = ImageEnhance.Contrast(bw_im)
contrast_im = contrast.enhance(10.0)

pix = contrast_im.load()
width, height = im.size

parsed_im_data = []

for y in range(0, height): 
    col = []
    for x in range(0, width): 
        current_pix = pix[x, y]
        if current_pix[0] <= PIX_THRESH:
            col.append((255, 255, 255))
        else: 
            col.append((0, 0, 0))

        # saturate, pick color 
    parsed_im_data.append(col)

f, axarray = plt.subplots(1, 3)

axarray[0].imshow(im)
axarray[0].title.set_text('Raw image')
axarray[1].imshow(contrast_im)
axarray[1].title.set_text('Processed image')
axarray[2].imshow(parsed_im_data) 
axarray[2].title.set_text('Extracted features')
plt.show()