import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

# Problem parameters
H = 1  # Pipe width (height)
L = 5  # Pipe length
T = 100  # Final time
aspect_ratio = H / L

S = 4e-2  # Sedimentation speed
u_in = 1.0  # Velocity inflow
u_crit = 1.2  # Critical velocity
phi_in = 0.2  # Volume fraction inflow
phi_c = 0.6  # Critical volume fraction

representative_G = 1.0  # Sedimentation parameter
representative_E = 1.0  # Erosion parameter
G_const = representative_G / (aspect_ratio * u_in)
E_const = representative_E / (aspect_ratio * u_in)

G_const = S
E_const = 0.9 * S

# Discretisation parameters
nt = 10000
nx = 200
ny = 40

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
h_max = np.zeros(nt)

for i in tqdm(range(1, nt)):
    # Solve for phi
    phi[i, :, :] = phi[i - 1, :, :]
    phi[i, 1:, 1:] += (
        - dt * u[i - 1, 1:, 1:] * (phi[i - 1, 1:, 1:] - phi[i - 1, 1:, :-1]) / dx
        - dt * phi[i - 1, 1:, 1:] * (u[i - 1, 1:, 1:] - u[i - 1, 1:, :-1]) / dx
        - dt * S * (phi[i - 1, 1:, 1:] - phi[i - 1, :-1, 1:]) / dy
    )
    
    j = np.ceil((1 - h[i - 1, :])/dy).astype(int)
    all_ = np.arange(nx)

    # Update sediment height
    h[i, :] = (
        h[i - 1, :]
        + dt * G_const * phi[i, j - 1, all_]
        - dt * E_const * np.maximum(u[i - 1, j - 1, all_]**2 - u_crit**2, 0)**1.5
    )

    h_max[i] = np.max(h[i, :])
    
    # Update velocity
    u[i, :, :] = 1/(1 - h[i, :])

    # Pressure drop
    p_drop[i-1] = np.trapezoid(0.026*100*(0.4*u[i, 0, :])**2/(2*10*(1-h[i - 1, :])), x=x)

fig, (ax, ax1) = plt.subplots(2, 1, figsize=(10, 8))

im = ax.imshow(phi[0, :, :], extent=[0, L, 0, H], vmin=np.min(phi), vmax=np.max(phi))
im2 = ax1.imshow(u[0, :, :], extent=[0, L, 0, H], vmin=np.min(u), vmax=np.max(u))

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
    im.set_data(phi[frame, :, :])
    im2.set_data(u[frame, :, :])
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

plt.plot(np.arange(0, nt)*dt*(100/0.4)/86400, h_max)
plt.xlabel("$t$ (Days)")
plt.ylabel(r"$h_{max}$")
plt.show()
