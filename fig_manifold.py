import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# --- Color definitions -------------------------------------------------------
# axis colors (bright)
col_real_x = '#FF0000'   # red
col_real_y = '#00FF00'   # green
col_real_z = '#0000FF'   # blue

col_imag_x = '#FFFF00'   # yellow
col_imag_y = '#00FFFF'   # cyan
col_imag_z = '#FF00FF'   # magenta

# label colors (darker / lower‑luminance versions for readability)
lab_real_x = '#880000'
lab_real_y = '#008800'
lab_real_z = '#000088'

lab_imag_x = '#999900'
lab_imag_y = '#009999'
lab_imag_z = '#990099'

# plane fill colours (light pastel versions)
plane_ab = '#FFFF99'   # light yellow
plane_cd = '#99FFFF'   # light cyan
plane_ef = '#FF99FF'   # light magenta

# --- Plot --------------------------------------------------------------------
fig = plt.figure(figsize=(8, 8), dpi=200)
ax = fig.add_subplot(111, projection='3d')
ax.set_proj_type('persp') # isometric view use 'ortho'

rng = np.linspace(-1, 1, 2)

# Plane (a,b): real‑x vs imag‑x  -> XY plane at z=0
X1, Y1 = np.meshgrid(rng, rng)
Z1 = np.zeros_like(X1)
ax.plot_surface(X1, Y1, Z1, alpha=0.3, color=plane_ab)

# Plane (c,d): real‑y vs imag‑y  -> YZ plane at x=0
Y2, Z2 = np.meshgrid(rng, rng)
X2 = np.zeros_like(Y2)
ax.plot_surface(X2, Y2, Z2, alpha=0.3, color=plane_cd)

# Plane (e,f): real‑z vs imag‑z  -> XZ plane at y=0
Z3, X3 = np.meshgrid(rng, rng)
Y3 = np.zeros_like(Z3)
ax.plot_surface(X3, Y3, Z3, alpha=0.3, color=plane_ef)

# Axis lines
ax.plot([-1, 1], [0, 0], [0, 0], color=col_real_x, linewidth=2)  # a
ax.plot([0, 0], [-1, 1], [0, 0], color=col_real_y, linewidth=2)  # c
ax.plot([0, 0], [0, 0], [-1, 1], color=col_real_z, linewidth=2)  # e

# Imaginary axis lines
ax.plot([1, 1], [0, 1], [0, 0], color=col_imag_x, linestyle='-', linewidth=1.2)  # b
ax.plot([0, 0], [1, 1], [0, 1], color=col_imag_y, linestyle='-', linewidth=1.2)  # d
ax.plot([0, 1], [0, 0], [1, 1], color=col_imag_z, linestyle='-', linewidth=1.2)  # f

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_cone(ax, base_center, direction, height=0.08, radius=0.03, color='black'):
    # Create a cone along the Z axis, then rotate it
    resolution = 20
    theta = np.linspace(0, 2 * np.pi, resolution)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.zeros_like(theta)

    tip = np.array([0, 0, height])
    base = np.stack((x, y, z), axis=1)

    # Construct faces
    faces = [[tip, base[i], base[(i + 1) % resolution]] for i in range(resolution)]

    # Translate and rotate to desired direction
    import numpy.linalg as la

    direction = np.array(direction)
    direction = direction / la.norm(direction)
    z_axis = np.array([0, 0, 1])

    v = np.cross(z_axis, direction)
    c = np.dot(z_axis, direction)
    if la.norm(v) < 1e-8:
        R = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (la.norm(v)**2))

    rotated_faces = [[[R @ point + base_center for point in tri] for tri in faces]]

    for face_set in rotated_faces:
        ax.add_collection3d(Poly3DCollection(face_set, color=color))

# Add cones at the positive ends of the real axes
draw_cone(ax, base_center=[1, 0, 0], direction=[1, 0, 0], color=col_real_x)
draw_cone(ax, base_center=[0, 1, 0], direction=[0, 1, 0], color=col_real_y)
draw_cone(ax, base_center=[0, 0, 1], direction=[0, 0, 1], color=col_real_z)
# Add cones at the positive ends of the imaginary axes
draw_cone(ax, base_center=[1, 1, 0], direction=[0, 1, 0], height=0.04, radius=0.015, color=col_imag_x)
draw_cone(ax, base_center=[0, 1, 1], direction=[0, 0, 1], height=0.04, radius=0.015, color=col_imag_y)
draw_cone(ax, base_center=[1, 0, 1], direction=[1, 0, 0], height=0.04, radius=0.015, color=col_imag_z)

# Imaginary axis lines (dashed for distinction)
ax.plot([0, 0], [0, 0], [0, 0], color='gray', linestyle='--', linewidth=1)  # placeholder (not used)

# Labels
ax.text(1.1, 0, 0, 'a (x‑real)', fontsize=9, color=lab_real_x)
ax.text(0, 1.1, 0, 'c (y‑real)', fontsize=9, color=lab_real_y)
ax.text(0, 0, 1.1, 'e (z‑real)', fontsize=9, color=lab_real_z)

ax.text(1, 1.1, 0, 'b (imaginary‑x)', fontsize=9, color=lab_imag_x, fontstyle='italic', fontfamily='serif')
ax.text(0, 1, 1.1, 'd (imaginary‑y)', fontsize=9, color=lab_imag_y, fontstyle='italic', fontfamily='serif')
ax.text(1.1, 0, 1, 'f (imaginary‑z)', fontsize=9, color=lab_imag_z, fontstyle='italic', fontfamily='serif')

# Aesthetic setup
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

ticks = np.linspace(-1, 1, 5)  # 5 ticks: -1, -0.5, 0, 0.5, 1

ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_zticks(ticks)

ax.tick_params(axis='x', labelsize=6)
ax.tick_params(axis='y', labelsize=6)
ax.tick_params(axis='z', labelsize=6)

#ax.set_xlabel('a / f')
#ax.set_ylabel('c / b')
#ax.set_zlabel('e / d')

# Center view
ax.view_init(elev=44, azim=-51) # isometric view (elev=35, azim=-45)
ax.set_box_aspect([1, 1, 1])

#ax.set_title('Three Complex Planes')

plt.tight_layout()

plt.show()
