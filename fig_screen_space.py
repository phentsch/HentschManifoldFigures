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
show_real_x = f # a
show_real_y = f # c
show_real_z = f # e
show_imag_x = f # b
show_imag_y = f # d
show_imag_z = f # f
show_plane_ab = f # e=0, d=0
show_plane_cd = f # a=0, f=0
show_plane_ef = f # c=0, b=0

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
figure_width = 8
figure_height = 8
figure_dpi = 300
#plt.tight_layout() # Adjust layout to prevent clipping of tick-labels
projection_type = 'persp' # 'persp' for perspective, 'ortho' for orthographic
    # viewing angle: for isometric view (elev=35, azim=-45)
elevation = 22 # elevation angle in degrees
azimuth = -39 # azimuth angle in degrees
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
u_vec = np.array([0, 0, 1.0])
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
        r'$\vec{\ell}$',
        fontsize=16,
        color=arrow_colour)


# --- Additional arrows: v₁ = (1,0,0), v₂ = (0,1,0) ---------------------------
# Arrow v₁ = (1, 0, 0)
ax.quiver(0, 0, 0, 1, 0, 0, color='#404040', arrow_length_ratio=0.1, linewidth=1)
ax.text(1.05, 0, 0, r'$\vec{v}_1$', fontsize=8, color='#404040')

# Arrow v₂ = (0, 1, 0)
ax.quiver(0, 0, 0, 0, 1, 0, color='#404040', arrow_length_ratio=0.1, linewidth=1)
ax.text(0, 1.05, 0, r'$\vec{v}_2$', fontsize=8, color='#404040')


# --- Rotated axes: α, β, γ through the origin (same rotation as cone) -------
# Compute rotation matrix that aligns v to z-axis
def rotation_matrix_from_vectors(vec1, vec2):
    a, b = vec1 / np.linalg.norm(vec1), vec2 / np.linalg.norm(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s < 1e-10:  # vectors are parallel
        return np.eye(3) if c > 0 else -np.eye(3)
    kmat = np.array([[ 0, -v[2], v[1]],
                     [ v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s**2))

# Define the vector v as a unit vector
v = np.array([1, 1, 1])
v = v / np.linalg.norm(v)

z_axis = np.array([0, 0, 1])  # Define z_axis as the unit vector along the z-axis
R = rotation_matrix_from_vectors(v, z_axis)
R_inv = np.linalg.inv(R)

rotated_vectors = R @ np.eye(3)  # columns: x, y, z mapped through forward rotation


# --- Rotated axes: α, β, γ as double-ended arrows through the origin -------
axis_length = 1
alpha = rotated_vectors[:, 0] * axis_length
beta = rotated_vectors[:, 1] * axis_length
gamma = rotated_vectors[:, 2] * axis_length

ax.quiver(*(-alpha), *(2 * alpha), color='magenta', arrow_length_ratio=0.025, linestyle=':', linewidth=1)
ax.text(*(alpha * 1.1), r'$\alpha$', fontsize=6, color='magenta')
ax.text(*(-alpha * 1.1), r'$-\alpha$', fontsize=6, color='magenta')

ax.quiver(*(-beta), *(2 * beta), color='#eeee00', arrow_length_ratio=0.025, linestyle=':', linewidth=1)
ax.text(*(beta * 1.1), r'$\beta$', fontsize=6, color='#eeee00')
ax.text(*(-beta * 1.1), r'$-\beta$', fontsize=6, color='#eeee00')

ax.quiver(*(-gamma), *(2 * gamma), color='cyan', arrow_length_ratio=0.025, linestyle=':', linewidth=1)
ax.text(*(gamma * 1.1), r'$\gamma$', fontsize=6, color='cyan')
ax.text(*(-gamma * 1.1), r'$-\gamma$', fontsize=6, color='cyan')


# --- Quadratic Null Cone Plot (Color-Coded and Normalized) -------------------

# Create 3D grid
x = np.linspace(-1, 1, 50)
y = np.linspace(-1, 1, 50)
z = np.linspace(-1, 1, 50)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')


# Normalize direction vector
v = np.array([1, 1, 1])
v = v / np.linalg.norm(v)
z_axis = np.array([0, 0, 1])

# Compute rotation matrix that aligns v to z-axis
def rotation_matrix_from_vectors(vec1, vec2):
    a, b = vec1 / np.linalg.norm(vec1), vec2 / np.linalg.norm(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s < 1e-10:  # vectors are parallel
        return np.eye(3) if c > 0 else -np.eye(3)
    kmat = np.array([[ 0, -v[2], v[1]],
                     [ v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s**2))

R = rotation_matrix_from_vectors(v, z_axis)
R_inv = np.linalg.inv(R)

# Rotate grid coordinates
X_flat = X.flatten()
Y_flat = Y.flatten()
Z_flat = Z.flatten()
coords = np.stack([X_flat, Y_flat, Z_flat], axis=1)
rotated_coords = coords @ R_inv.T
Xr, Yr, Zr = rotated_coords[:,0].reshape(X.shape), rotated_coords[:,1].reshape(Y.shape), rotated_coords[:,2].reshape(Z.shape)

# Evaluate the original function in rotated coordinates
F = Xr * Yr + Yr * Zr + Zr * Xr

# Use marching cubes to extract the isosurface f=0
verts, faces, normals, values = marching_cubes(F, level=0, spacing=(x[1]-x[0], y[1]-y[0], z[1]-z[0]))
verts += np.array([x[0], y[0], z[0]])

 # Compute color values based on z-coordinate for color mapping
proj = verts[:, 2]  # use z-coordinate directly for color mapping
proj_normalized = (proj - proj.min()) / (proj.max() - proj.min())  # normalize to [0,1]

# Create and add surface with color mapping
mesh = Poly3DCollection(verts[faces], alpha=0.2, edgecolor='k', linewidths=0.05)
mesh.set_array(proj_normalized[faces].mean(axis=1))  # average color per face
mesh.set_cmap('turbo')
ax.add_collection3d(mesh)

#fig.colorbar(mesh, ax=ax, shrink=0.6, aspect=10)


# --- Add screen plane orthogonal to ℓ = (1,1,1) -------------------------------
# Define horizontal screen plane at z = 0.5 (orthogonal to ℓ aligned to z-axis)

ell_value = 0.575
screen_range = np.linspace(-1, 1, 2)
X_screen, Y_screen = np.meshgrid(screen_range, screen_range)
Z_screen = np.full_like(X_screen, ell_value)

# Color the circle segments according to z value using turbo colormap and normalization
from matplotlib.cm import get_cmap
cmap = plt.cm.turbo
norm = plt.Normalize(vmin=-1, vmax=1)

screen_color = cmap(norm(ell_value))
ax.plot_surface(X_screen, Y_screen, Z_screen, alpha=0.2, color=screen_color, edgecolor='none')
ax.text(-0.96, -0.96, ell_value, r'$\ell^\perp$', fontsize=10, color=screen_color)

# Add a circular null cone cross-section on the screen plane
circle_radius = np.sqrt(2)*(ell_value)  # approximate radius; adjust as needed
theta = np.linspace(0, 2 * np.pi, 100)
x_circle = circle_radius * np.cos(theta)
y_circle = circle_radius * np.sin(theta)
z_circle = np.full_like(x_circle, ell_value)

for i in range(len(theta) - 1):
    x_seg = [x_circle[i], x_circle[i+1]]
    y_seg = [y_circle[i], y_circle[i+1]]
    z_seg = [z_circle[i], z_circle[i+1]]
    val = 0.5 * (z_circle[i] + z_circle[i+1])  # average z value for color
    ax.plot(x_seg, y_seg, z_seg, color=cmap(norm(val)), linewidth=1.2)

val_label = ell_value+0.2
label_color = cmap(norm(val_label))
ax.text(0, -1.6, ell_value, r'$\mathcal{S}_P = T_P\mathcal{M}_{\text{H}} / \text{span}(\ell)$', fontsize=8, color=label_color)
ax.text(0.15, 0.15, ell_value, 'sub-null', fontfamily='serif', fontstyle='italic', fontsize=7, color=label_color)
ax.text(0.65, 0.65, ell_value, 'super-null', fontfamily='serif', fontstyle='italic', fontsize=7, color=label_color)


# --- End of Additional Plots ---------------------------------------------

# --- Show the plot ------------------------------------------------
plt.show()
# Save the figure
# fig.savefig('Figure_4.png', dpi=figure_dpi, bbox_inches='tight')
