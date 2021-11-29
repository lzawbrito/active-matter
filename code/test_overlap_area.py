from src.mathutils import rectangle_overlap, area
import matplotlib.pyplot as plt 
import numpy as np

a_verts = [[0, 0], [0, 4], [2, 4], [2, 0]]
b_verts = [[1, 1], [1, 2], [4, 2], [4, 1]]

print(rectangle_overlap(a_verts, b_verts))

a_verts = [[0, 0], [0, 4], [2, 4], [2, 0]]
b_verts = [[0.5, 1], [0.5, 2], [1.5, 2], [1.5, 1]]

print(rectangle_overlap(a_verts, b_verts))

a_verts = [[0, 0], [0, 2], [2, 2], [2, -2]]
b_verts = [[-1, -1], [4, 4], [2, 6], [-3, 1]]
print(rectangle_overlap(a_verts, b_verts))

a_verts = [[0, 0], [0, 2], [2, 2], [2, 0]]
b_verts = [[0.25, 0.25], [1.75, 1.75], [3.25, 0.25], [1.75, -1.25]]
print(rectangle_overlap(a_verts, b_verts))
print(area([[-1, -1], [-1, 0.5], [-0.75, 0.75], [0.75, -0.75], [0.5, -1]]))

a_verts.append(a_verts[0])
b_verts.append(b_verts[0])
ax, ay = np.transpose(a_verts)
bx, by = np.transpose(b_verts)

plt.plot(ax, ay, color='red')
plt.plot(bx, by, color='red')

plt.show()


