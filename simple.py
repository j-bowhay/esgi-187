import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from tqdm import tqdm


u = 4 # this will become non-constant
s = 0.8

phi_in = 0.2
phi_c = 0.6

nt = 4000
nx = 200
ny = 40

H = 1
L = 10
T = 10

dx = L / nx
dy = H / ny
dt = T / nt

print(f"udt/dx = {u*dt/dx}, s*dt/dy = {s*dt/dy}")


phi = np.zeros((nt, ny, nx))
phi[:, 0, :] = 0 # Boundary condition at the top edge
phi[:, :, 0] = phi_in  # Boundary condition at the left edge

h = np.zeros((nt, nx))

for i in tqdm(range(1, nt)):
    phi[i, :, :] = phi[i - 1, :, :]
    phi[i, 1:, 1:] += - (u*dt/dx) *(phi[i - 1, 1:, 1:] - phi[i - 1, 1:, :-1])
    phi[i, 1:, 1:] += (s*dt/dy) * (phi[i - 1, :-1, 1:] - phi[i - 1, 1:, 1:])
    h[i, :] = h[i - 1, :] + s*dt * phi[i-1, -1, :]/(phi_c - phi[i-1, -1, :])

fig, (ax, ax1) = plt.subplots(2, 1, figsize=(10, 8))

im = ax.imshow(phi[0, :, :], extent=[0, L, 0, H])
im2 = ax1.imshow(u*np.ones_like(phi[0, :, :]), extent=[0, L, 0, H])
line = ax.plot(np.linspace(0, L, nx), h[0, :], color='red')
line2 = ax1.plot(np.linspace(0, L, nx), h[0, :], color='red')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_title("$u$")

def update(frame):
    im.set_array(phi[frame, :, :])
    ax.set_title(f"Time: {frame * dt:.2f} seconds\n$\\phi$")
    line[0].set_ydata(h[frame, :])
    line2[0].set_ydata(h[frame, :])

ani = animation.FuncAnimation(plt.gcf(), update, frames=nt, interval=50)
plt.show()