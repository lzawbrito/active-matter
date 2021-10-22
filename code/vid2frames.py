import av

VIDEO = './data/videos/10-22-2021/test.MOV'

container = av.open(VIDEO)

for frame in container.decode(video=0):
    frame.to_image().save('frame-%04d.jpg' % frame.index)