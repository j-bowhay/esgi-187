import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from tqdm import tqdm


s = 0.05

phi_in = 0.2
phi_c = 0.6

ntau = 10000
nX = 200
nxi = 40

H = 1
L = 10
T = 19

dX = L / nX
dxi = H / nxi
dtau = T / ntau


phi = np.zeros((ntau, nxi, nX))
phi[:, :, 0] = phi_in  # Boundary condition at the left edge
phi[:, 0, :] = 0 # Boundary condition at the top edge

u = np.ones((ntau, nxi, nX))

print(f"udt/dx = {np.max(u)*dtau/dX}, s*dt/dy = {s*dtau/dxi}")

h = np.zeros((ntau, nX))

for i in tqdm(range(1, ntau)):
    # previous time step
    phi[i, :, :] = phi[i - 1, :, :]
    # -u phi_x
    phi[i, 1:, 1:] += - (u[i - 1, 1:, 1:]*dtau/dX) *(phi[i - 1, 1:, 1:] - phi[i - 1, 1:, :-1])
    # -u_x phi
    phi[i, 1:, 1:] += - (phi[i - 1, 1:, 1:]*dtau/dX) *(u[i - 1, 1:, 1:] - u[i - 1, 1:, :-1])
    # +s/(1-h) phi_y
    phi[i, 1:, 1:] += (s*dtau/(dxi*(1-h[i-1, 1:]))) * (phi[i - 1, :-1, 1:] - phi[i - 1, 1:, 1:])
    # + D/(1-h)^2 phi_xixi
    phi[i, 1:, 1:-1] += (0.01*dtau/(dxi**2*(1-h[i-1, 1:-1])**2)) * (phi[i - 1, 1:, 2:] - 2*phi[i - 1, 1:, 1:-1] + phi[i - 1, 1:, :-2])
    # todo:
    # - add boundary condition at the bottom for phi
    # - properly set the diffusion coefficient
    # - double check the if the equation for updating h needs to be changed
    # - add the h_x / h_t terms
    # (phi_c - phi) h_t = J.n
    h[i, :] = h[i - 1, :] + s*dtau * phi[i-1, -1, :]/(phi_c - phi[i-1, -1, :])
    # u = 1/1 - h
    u[i, :, :] = 1/(1 - h[i, :])

fig, (ax, ax1) = plt.subplots(2, 1, figsize=(12, 4), tight_layout=True)

xi = np.linspace(0, 1, nxi)
X = np.linspace(0, L, nX)
X, xi = np.meshgrid(X, xi[::-1])
umax = np.max(u)

phi_plot = ax.pcolormesh(X, h[0, :] + (1 - h[0, :])*xi, phi[0, :, :], clim=(0, phi_in))
u_plot = ax1.pcolormesh(X, h[0, :] + (1 - h[0, :])*xi, u[0, :, :], clim=(0, umax))
ax.set_xlabel("x")
ax.set_ylabel("y")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_title("$u$")
ax1.set_ylim(0, 1)
ax.set_ylim(0, 1)
fig.colorbar(phi_plot, ax=ax, label="$\\phi$")
fig.colorbar(u_plot, ax=ax1, label="$u$")

def update(frame):
    global phi_plot, u_plot
    phi_plot.remove()
    phi_plot = ax.pcolormesh(X, h[frame, :] + (1 - h[frame, :])*xi, phi[frame, :, :], clim=(0, phi_in))
    u_plot.remove()
    u_plot = ax1.pcolormesh(X, h[frame, :] + (1 - h[frame, :])*xi, u[frame, :, :], clim=(0, umax))
    ax.set_title(f"Time: {frame * dtau:.2f}\n$\\phi$")

ani = animation.FuncAnimation(fig, update, frames=np.arange(0, ntau, 100), interval=1)
plt.show()