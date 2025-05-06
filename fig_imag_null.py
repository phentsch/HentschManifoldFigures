import numpy as np
import matplotlib.pyplot as plt # Import matplotlib for plotting
from mpl_toolkits.mplot3d import Axes3D # Import 3D plotting tools
from mpl_toolkits.mplot3d.art3d import Poly3DCollection # Import Poly3DCollection for 3D polygon collection
from skimage.measure import marching_cubes # Import marching_cubes for isosurface extraction

# --- Color Definitions -------------------------------------------------------
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
    # plane fill colours (muted versions)
plane_ab = '#AAAA00'   # yellow
plane_cd = '#00AAAA'   # cyan
plane_ef = '#AA00AA'   # magenta

# --- Axis Toggles ------------------------------------------------
f = False
t = True
show_real_x = t # a
show_real_y = t # c
show_real_z = t # e
show_imag_x = t # b
show_imag_y = t # d
show_imag_z = t # f
show_plane_ab = t # e=0, d=0
show_plane_cd = t # a=0, f=0
show_plane_ef = t # c=0, b=0

# --- Customizations ------------------------------------------------
    # plot title
# Dynamically construct plot title from active axes
subspace_axes = []
if show_real_x: subspace_axes.append('a')
if show_real_y: subspace_axes.append('c')
if show_real_z: subspace_axes.append('e')
if show_imag_x: subspace_axes.append('b')
if show_imag_y: subspace_axes.append('d')
if show_imag_z: subspace_axes.append('f')
plot_title = f"3D Subspace {{{', '.join(subspace_axes)}}}"
title_size = 9 # title font size
title_style = 'normal' # title font style
title_family = 'serif' # title font family
title_location = 'center' # title location
title_padding = 25 # title padding
    # figure setup
figure_width = 6
figure_height = 6
figure_dpi = 200
#plt.tight_layout() # Adjust layout to prevent clipping of tick-labels
projection_type = 'persp' # 'persp' for perspective, 'ortho' for orthographic
    # viewing angle: for isometric view (elev=35, azim=-45)
elevation = 20 # elevation angle in degrees
azimuth = -50 # azimuth angle in degrees
    # resolution of the plot
rng = np.linspace(-1, 1, 2) # rng (range)
    # bounds
x_bound_neg = -1
x_bound_pos = 1
y_bound_neg = -1
y_bound_pos = 1
z_bound_neg = -1
z_bound_pos = 1
    # aspect ratio
aspect_ratio = [1, 1, 1] # [x, y, z]
    # ticks for the axes
ticks = np.linspace(-1, 1, 5)
tick_size = 5 # tick label font size
    # axis labels
x_label_parts = []
if show_real_x: x_label_parts.append(r'$\alpha$')
if show_imag_z: x_label_parts.append(r'$\gamma$')
x_label = ', '.join(x_label_parts) if x_label_parts else '$x$'

y_label_parts = []
if show_real_y: y_label_parts.append(r'$\beta$')
if show_imag_x: y_label_parts.append(r'$\alpha$')
y_label = ', '.join(y_label_parts) if y_label_parts else '$y$'

z_label_parts = []
if show_real_z: z_label_parts.append(r'$\gamma$')
if show_imag_y: z_label_parts.append(r'$\beta$')
z_label = ', '.join(z_label_parts) if z_label_parts else '$z$'
label_size = 6 # axis label font size
    # axes styling
rlw = 2 # real axis line width
rls = '-' # real axis line style
ilw = 2 # imaginary axis line width
ils = ':' # imaginary axis line style
    # axis labels styling
rfsize = 9 # real axis font size
rfstyle = 'normal' # real axis font style
rffamily = 'sans' # real axis font family
ifsize = 9 # imaginary axis font size
ifstyle = 'italic' # imaginary axis font style
iffamily = 'serif' # imaginary axis font family
    # axis arrow cones
arrow_height = 0.075
arrow_radius = 0.02
    # plane alpha (transparency)
p_alpha = 0.1

# --- Construction of Axis Arrows (Cones) ---------------------------------------------
def draw_cone(ax, base_center, direction, height=arrow_height, radius=arrow_radius, color='black'):
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

# --- Plot --------------------------------------------------------------------
fig = plt.figure(figsize=(figure_width, figure_height), dpi=figure_dpi)
ax = fig.add_subplot(111, projection='3d')
ax.set_proj_type(projection_type)
ax.set_xlim(x_bound_neg, x_bound_pos)
ax.set_ylim(y_bound_neg, y_bound_pos)
ax.set_zlim(z_bound_neg, z_bound_pos)
ax.set_box_aspect(aspect_ratio)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_zticks(ticks)
ax.tick_params(axis='x', labelsize=tick_size)
ax.tick_params(axis='y', labelsize=tick_size)
ax.tick_params(axis='z', labelsize=tick_size)
ax.set_xlabel(x_label, fontsize=label_size)
ax.set_ylabel(y_label, fontsize=label_size)
ax.set_zlabel(z_label, fontsize=label_size)
ax.view_init(elev=elevation, azim=azimuth)
#ax.set_title(plot_title, fontsize=title_size, fontstyle=title_style, fontfamily=title_family, loc=title_location, pad=title_padding)

# --- Axis lines (a, b, c, d, e, f) --------------------------------
# Real Axis lines (a, c, e)
if show_real_x:
    ax.plot([-1, 1], [0, 0], [0, 0], color=col_real_x, linestyle=rls, linewidth=rlw)  # a
    draw_cone(ax, base_center=[1, 0, 0], direction=[1, 0, 0], color=col_real_x)
    ax.text(1.1, 0, 0, r'$\alpha$', fontsize=rfsize, color=lab_real_x, fontstyle=rfstyle, fontfamily=rffamily)
if show_real_y:
    ax.plot([0, 0], [-1, 1], [0, 0], color=col_real_y, linestyle=rls, linewidth=rlw)  # c
    draw_cone(ax, base_center=[0, 1, 0], direction=[0, 1, 0], color=col_real_y)
    ax.text(0, 1.1, 0, r'$\beta$', fontsize=rfsize, color=lab_real_y, fontstyle=rfstyle, fontfamily=rffamily)
if show_real_z:
    ax.plot([0, 0], [0, 0], [-1, 1], color=col_real_z, linestyle=rls, linewidth=rlw)  # e
    draw_cone(ax, base_center=[0, 0, 1], direction=[0, 0, 1], color=col_real_z)
    ax.text(0, 0, 1.1, r'$\gamma$', fontsize=rfsize, color=lab_real_z, fontstyle=rfstyle, fontfamily=rffamily)
# Imaginary axis lines (b, d, f)
if show_imag_x:
    ax.plot([0, 0], [-1, 1], [0, 0], color=col_imag_x, linestyle=ils, linewidth=ilw)  # b
    draw_cone(ax, base_center=[0, 1, 0], direction=[0, 1, 0], color=col_imag_x)
    ax.text(0, 1.2, 0, r'$\alpha$', fontsize=ifsize, color=lab_imag_x, fontstyle=ifstyle, fontfamily=iffamily)
if show_imag_y:
    ax.plot([0, 0], [0, 0], [-1, 1], color=col_imag_y, linestyle=ils, linewidth=ilw)  # d
    draw_cone(ax, base_center=[0, 0, 1], direction=[0, 0, 1], color=col_imag_y)
    ax.text(0, 0, 1.2, r'$\beta$', fontsize=ifsize, color=lab_imag_y, fontstyle=ifstyle, fontfamily=iffamily)
if show_imag_z:
    ax.plot([-1, 1], [0, 0], [0, 0], color=col_imag_z, linestyle=ils, linewidth=ilw)  # f
    draw_cone(ax, base_center=[1, 0, 0], direction=[1, 0, 0], color=col_imag_z)
    ax.text(1.2, 0, 0, r'$\gamma$', fontsize=ifsize, color=lab_imag_z, fontstyle=ifstyle, fontfamily=iffamily)

# --- Complex planes (a,b), (c,d), (e,f) -----------------------------
#    # Plane (a,b): real‑x vs imag‑x  -> XY plane at z=0
if show_plane_ab:
    X1, Y1 = np.meshgrid(rng, rng)
    Z1 = np.zeros_like(X1)
    ax.plot_surface(X1, Y1, Z1, alpha=p_alpha, color=plane_ab)
#    # Plane (c,d): real‑y vs imag‑y  -> YZ plane at x=0
if show_plane_cd:
    Y2, Z2 = np.meshgrid(rng, rng)
    X2 = np.zeros_like(Y2)
    ax.plot_surface(X2, Y2, Z2, alpha=p_alpha, color=plane_cd)
#    # Plane (e,f): real‑z vs imag‑z  -> XZ plane at y=0
if show_plane_ef:
    Z3, X3 = np.meshgrid(rng, rng)
    Y3 = np.zeros_like(Z3)
    ax.plot_surface(X3, Y3, Z3, alpha=p_alpha, color=plane_ef)

# --- Additional Plots ------------------------------------------------

# --- Central (1,1,1) direction line -----------------------------------------
# Unit vector along (1,1,1)
u_vec = np.array([1.0, 1.0, 1.0])
u_vec = u_vec / np.linalg.norm(u_vec)

# Parameter t ∈ [‑1,1] so that the segment length is 2
t_vals = np.linspace(-1.0, 1.0, 150)
line_pts = np.outer(t_vals, u_vec)                # shape (150,3)
a_line, c_line, e_line = line_pts[:,0], line_pts[:,1], line_pts[:,2]

# Colour values using the *same* normalisation as the scatter plot
colour_raw_line = a_line + c_line + e_line
color_min = colour_raw_line.min()  # Compute color_min
color_max = colour_raw_line.max()  # Compute color_max
colour_norm_line = (colour_raw_line - color_min) / (color_max - color_min)

# Plot the line as a dense scatter cloud so it inherits the colormap gradient
ax.scatter(a_line, c_line, e_line,
           c=colour_norm_line,
           cmap='turbo',
           s=1.5,
           alpha=1)

# Arrowhead in the colour corresponding to the positive end
arrow_colour = plt.cm.turbo((colour_raw_line[-1] - color_min) / (color_max - color_min))
draw_cone(ax,
          base_center=line_pts[-1],
          direction=u_vec,
          height=arrow_height,
          radius=arrow_radius,
          color=arrow_colour)

# Label for the direction vector ℓ
ax.text(line_pts[-1,0] + 0.05,
        line_pts[-1,1] + 0.05,
        line_pts[-1,2] + 0.05,
        r'$\ell$',
        fontsize=8,
        color=arrow_colour)


# --- Quadratic Null Cone Plot (Color-Coded and Normalized) -------------------

# Create 3D grid
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
z = np.linspace(-1, 1, 100)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Define the implicit function f(x, y, z) = xy + yz + zx
F = X*(Y) + Y*(Z) + Z*(X)

# Use marching cubes to extract the isosurface f=0
verts, faces, normals, values = marching_cubes(F, level=0, spacing=(x[1]-x[0], y[1]-y[0], z[1]-z[0]))
verts += np.array([x[0], y[0], z[0]])

# Compute color values based on projection along (1,1,1)
proj = verts @ np.array([1, 1, 1]) / np.sqrt(3)  # normalize by sqrt(3) to keep scale reasonable
proj_normalized = (proj - proj.min()) / (proj.max() - proj.min())  # normalize to [0,1]

# Create and add surface with color mapping
mesh = Poly3DCollection(verts[faces], alpha=0.5, edgecolor='k', linewidths=0.05)
mesh.set_array(proj_normalized[faces].mean(axis=1))  # average color per face
mesh.set_cmap('turbo')
ax.add_collection3d(mesh)

#fig.colorbar(mesh, ax=ax, shrink=0.6, aspect=10)


# --- End of Additional Plots ---------------------------------------------

# --- Show the plot ------------------------------------------------
plt.show()
