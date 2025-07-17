import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from tqdm import tqdm


S = 4e-2

phi_in = 0.2
phi_c = 0.6

nt = 10000
nx = 200
ny = 40

H = 1
L = 1
T = 19

dx = L / nx
dy = H / ny
dt = T / nt


phi = np.zeros((nt, ny, nx))
phi[:, :, 0] = phi_in  # Boundary condition at the left edge
phi[:, 0, :] = 0 # Boundary condition at the top edge

u = np.ones((nt, ny, nx))

p_drop = np.ones(nt-1)
x = np.linspace(0, L, nx)

print(f"udt/dx = {np.max(u)*dt/dx}, s*dt/dy = {S*dt/dy}")

h = np.zeros((nt, nx))

for i in tqdm(range(1, nt)):
    phi[i, :, :] = phi[i - 1, :, :]
    phi[i, 1:, 1:] += - (u[i - 1, 1:, 1:]*dt/dx) *(phi[i - 1, 1:, 1:] - phi[i - 1, 1:, :-1])
    phi[i, 1:, 1:] += - (phi[i - 1, 1:, 1:]*dt/dx) *(u[i - 1, 1:, 1:] - u[i - 1, 1:, :-1])
    phi[i, 1:, 1:] += (S*dt/dy) * (phi[i - 1, :-1, 1:] - phi[i - 1, 1:, 1:])
    j = np.ceil((1 - h[i - 1, :])/dy).astype(int)
    all_ = np.arange(nx)
    h[i, :] = h[i - 1, :] + S*dt * phi[i-1, j-1, all_]/(phi_c - phi[i-1, j-1, all_])
    u[i, :, :] = 1/(1 - h[i, :])
    p_drop[i-1] = np.trapezoid(0.026*100*(0.4*u[i, 0, :])**2/(2*10*(1-h[i - 1, :])), x=x)

fig, (ax, ax1) = plt.subplots(2, 1, figsize=(10, 8))

im = ax.imshow(phi[0, :, :], extent=[0, L, 0, H])
im2 = ax1.imshow(u[int(0.7*nt)], extent=[0, L, 0, H])
im2.set_array(u[0, :, :])
fill = ax.fill_between(np.linspace(0, L, nx), 0, h[0, :], color='red')
fill2 = ax1.fill_between(np.linspace(0, L, nx), 0, h[0, :], color='red')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_title("$u$")
fig.colorbar(im, ax=ax, label="$\\phi$")
fig.colorbar(im2, ax=ax1, label="$u$")

def update(frame):
    global fill, fill2
    im.set_array(phi[frame, :, :])
    im2.set_array(u[frame, :, :])
    ax.set_title(f"Time: {frame * dt:.2f}\n$\\phi$")
    fill.remove()
    fill = ax.fill_between(np.linspace(0, L, nx), 0, h[frame, :], color='red')
    fill2.remove()
    fill2 = ax1.fill_between(np.linspace(0, L, nx), 0, h[frame, :], color='red')

ani = animation.FuncAnimation(fig, update, frames=np.arange(0, nt, 50), interval=1)
plt.show()

plt.plot(np.arange(0, nt-1)*dt*(100/0.4)/86400, p_drop)
plt.xlabel("$t (Days)$")
plt.ylabel(r"$\Delta p$")
plt.show()