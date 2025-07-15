import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from tqdm import tqdm


u = 1 # this will become non-constant
s = 0.8

nt = 500
nx = 100
ny = 25

H = 1
L = 10
T = 10

dx = L / nx
dy = H / ny
dt = T / nt

print(f"udt/dx = {u*dt/dx}, s*dt/dy = {s*dt/dy}")


phi = np.zeros((nt, ny, nx))
phi[:, 0, :] = 0 # Boundary condition at the top edge
phi[:, :, 0] = 0.5  # Boundary condition at the left edge

for i in tqdm(range(1, nt)):
    phi[i, :, :] = phi[i - 1, :, :]
    phi[i, 1:, 1:] += - (u*dt/dx) *(phi[i - 1, 1:, 1:] - phi[i - 1, 1:, :-1])
    phi[i, 1:, 1:] += (s*dt/dy) * (phi[i - 1, :-1, 1:] - phi[i - 1, 1:, 1:])

im = plt.imshow(phi[0, :, :], )

def update(frame):
    im.set_array(phi[frame, :, :])
    plt.title(f"Time: {frame * dt:.2f} seconds")

ani = animation.FuncAnimation(plt.gcf(), update, frames=nt, interval=50)
plt.show()