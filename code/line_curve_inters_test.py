from src.mathutils import align_rectangles, line_seg_intersection, \
		vertices_inside, area, InfinitelyManySolutionsError, \
		NoSolutionError
import matplotlib.pyplot as plt 
import numpy as np 

inter = line_seg_intersection([1, 1], [1, -1], [-1, 0], [2, 0])
print(inter)

try:
	line_seg_intersection([-2, 0], [3, 0], [-1, 0], [1, 0])
except InfinitelyManySolutionsError as e:
	print('Caught exception: ' + e.__str__())

try:
	line_seg_intersection([-2, -1], [-3, -1], [1, 4], [2, 5])
except NoSolutionError as e:
	print('Caught exception: ' + e.__str__())

try:
	line_seg_intersection([0, -1], [0, 1], [1, 1], [1, -1])
except NoSolutionError as e:
	print('Caught exception: ' + e.__str__())


a_verts = [[0, 0], [0, 2], [2, 2], [2, 0]]
b_verts = [[0.25, 0.25], [1.75, 1.75], [3.25, 0.25], [1.75, -1.25]]
# ^ Further up in original coordinate system
# print(vertices_inside(b_verts, a_verts))

new_a_verts, new_b_verts = align_rectangles(a_verts, b_verts)

a_verts.append(a_verts[0])
b_verts.append(b_verts[0])
new_a_verts.append(new_a_verts[0])
new_b_verts.append(new_b_verts[0])

ax, ay = np.transpose(a_verts)
bx, by = np.transpose(b_verts)
new_ax, new_ay = np.transpose(new_a_verts)
new_bx, new_by = np.transpose(new_b_verts)

plt.plot(ax, ay, color='red')
plt.plot(bx, by, color='red')
plt.plot(new_ax, new_ay, color='blue')
plt.plot(new_bx, new_by, color='blue')

plt.show()



