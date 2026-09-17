import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon
import numpy as np

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "mathtext.fontset": "stix",
})

# ---------- Figure 3.1 ----------
fig, ax = plt.subplots(figsize=(6.2, 5.6))
ax.set_xlim(-0.6, 1.6)
ax.set_ylim(-0.6, 1.6)
ax.set_aspect("equal")
ax.axhline(0, color="0.75", lw=0.8)
ax.axvline(0, color="0.75", lw=0.8)
ax.plot([0.5, 0.5], [-0.6, 1.6], color="#c0392b", lw=2.2, zorder=2, label=r"$x_1=1/2$")
ax.fill_betweenx([-0.6, 1.6], -0.6, 0.5, color="#3498db", alpha=0.12)
ax.fill_betweenx([-0.6, 1.6], 0.5, 1.6, color="#e67e22", alpha=0.12)
ax.scatter([0, 0], [0, 1], s=90, c="#2980b9", zorder=3, label=r"$\omega_1$")
ax.scatter([1, 1], [0, 1], s=90, c="#d35400", marker="s", zorder=3, label=r"$\omega_2$")
for xy, txt, dx, dy in [
    ((0, 0), r"$[0,0]^\mathrm{T}$", -0.28, -0.18),
    ((0, 1), r"$[0,1]^\mathrm{T}$", -0.28, 0.08),
    ((1, 0), r"$[1,0]^\mathrm{T}$", 0.08, -0.18),
    ((1, 1), r"$[1,1]^\mathrm{T}$", 0.08, 0.08),
]:
    ax.annotate(txt, xy, xytext=(xy[0] + dx, xy[1] + dy), fontsize=10)
ax.text(0.05, 1.42, r"$\omega_1$: $w^\mathrm{T}y>0$", color="#1a5276", fontsize=11)
ax.text(0.95, 1.42, r"$\omega_2$: $w^\mathrm{T}y<0$", color="#6e2c00", fontsize=11)
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_title(r"Problem 3.1: perceptron decision line $x_1=1/2$")
ax.legend(loc="lower right")
ax.set_xticks([0, 0.5, 1])
ax.set_yticks([0, 1])
fig.tight_layout()
fig.savefig("fig_3_1.png", dpi=180, bbox_inches="tight")
plt.close()

# ---------- Figure 3.2: three lines and regions ----------
fig, ax = plt.subplots(figsize=(7.4, 7.0))
xs = np.linspace(-2.2, 2.2, 400)
ax.plot(xs, -xs, color="#1f77b4", lw=2, label=r"$g_1: x_1+x_2=0$")
ax.axhline(0.25, color="#d62728", lw=2, label=r"$g_2: x_2=1/4$")
ax.plot(xs, xs, color="#2ca02c", lw=2, label=r"$g_3: x_1-x_2=0$")

# triangle
tri = np.array([[-0.25, 0.25], [0.0, 0.0], [0.25, 0.25]])
ax.add_patch(Polygon(tri, closed=True, facecolor="#f4d03f", edgecolor="none", alpha=0.45, zorder=0))

labels = [
    (0.0, 0.12, r"$R_1$ $(+,-,-)$" + "\n" + r"$(1,0,0)$"),
    (0.0, 1.15, r"$R_2$ $(+,+,-)$" + "\n" + r"$(1,1,0)$"),
    (-1.25, 1.15, r"$R_3$ $(-,+,-)$" + "\n" + r"$(0,1,0)$"),
    (1.25, 1.15, r"$R_4$ $(+,+,+)$" + "\n" + r"$(1,1,1)$"),
    (1.15, -0.15, r"$R_5$ $(+,-,+)$" + "\n" + r"$(1,0,1)$"),
    (-1.15, -0.15, r"$R_6$ $(-,-,-)$" + "\n" + r"$(0,0,0)$"),
    (0.0, -1.15, r"$R_7$ $(-,-,+)$" + "\n" + r"$(0,0,1)$"),
]
for x, y, t in labels:
    ax.text(x, y, t, ha="center", va="center", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.92))

ax.scatter([-0.25, 0, 0.25], [0.25, 0, 0.25], c="k", s=22, zorder=4)
ax.annotate(r"$(-\frac{1}{4},\frac{1}{4})$", (-0.25, 0.25), xytext=(-1.05, 0.55),
            fontsize=9, arrowprops=dict(arrowstyle="->", color="0.3"))
ax.annotate(r"$(0,0)$", (0, 0), xytext=(0.35, -0.55),
            fontsize=9, arrowprops=dict(arrowstyle="->", color="0.3"))
ax.annotate(r"$(\frac{1}{4},\frac{1}{4})$", (0.25, 0.25), xytext=(0.7, 0.55),
            fontsize=9, arrowprops=dict(arrowstyle="->", color="0.3"))

ax.set_xlim(-2.05, 2.05)
ax.set_ylim(-1.85, 1.85)
ax.set_aspect("equal")
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_title("Problem 3.2: three lines and the 7 polyhedral regions")
ax.legend(loc="lower left", fontsize=9)
ax.axhline(0, color="0.8", lw=0.7)
ax.axvline(0, color="0.8", lw=0.7)
fig.tight_layout()
fig.savefig("fig_3_2_regions.png", dpi=180, bbox_inches="tight")
plt.close()

# ---------- Cube vertices ----------
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

fig = plt.figure(figsize=(8.2, 6.4))
ax = fig.add_subplot(111, projection="3d")

verts = {
    (0, 0, 0): r"$R_6$",
    (1, 0, 0): r"$R_1$",
    (0, 1, 0): r"$R_3$",
    (1, 1, 0): r"$R_2$",
    (0, 0, 1): r"$R_7$",
    (1, 0, 1): r"$R_5$",
    (1, 1, 1): r"$R_4$",
}
# draw cube edges
edges = [
    ((0, 0, 0), (1, 0, 0)), ((0, 0, 0), (0, 1, 0)), ((0, 0, 0), (0, 0, 1)),
    ((1, 1, 0), (0, 1, 0)), ((1, 1, 0), (1, 0, 0)), ((1, 1, 0), (1, 1, 1)),
    ((0, 1, 1), (0, 0, 1)), ((0, 1, 1), (0, 1, 0)), ((0, 1, 1), (1, 1, 1)),
    ((1, 0, 1), (0, 0, 1)), ((1, 0, 1), (1, 0, 0)), ((1, 0, 1), (1, 1, 1)),
]
for a, b in edges:
    ax.plot(*zip(a, b), color="0.55", lw=1.2)
# missing vertex (0,1,1) as hollow
ax.scatter([0], [1], [1], s=80, facecolors="none", edgecolors="#7f8c8d", linewidths=1.6, depthshade=False)
ax.text(0.02, 1.08, 1.08, r"unused $(0,1,1)$", fontsize=8, color="0.4")

for (y1, y2, y3), lab in verts.items():
    ax.scatter([y1], [y2], [y3], s=55, c="#1f77b4", depthshade=False)
    ax.text(y1 + 0.05, y2 + 0.05, y3 + 0.06, f"{lab} ({y1},{y2},{y3})", fontsize=8)

ax.set_xlabel(r"$y_1$")
ax.set_ylabel(r"$y_2$")
ax.set_zlabel(r"$y_3$")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_zticks([0, 1])
ax.set_title("First-layer mapping onto vertices of the unit cube")
ax.view_init(elev=18, azim=-62)
fig.tight_layout()
fig.savefig("fig_3_2_cube.png", dpi=180, bbox_inches="tight")
plt.close()

print("saved figures")
