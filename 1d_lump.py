import numpy as np

import matplotlib.animation as animation

L = 1
nx = 200

dx = L / (nx - 1)

u_crit = 0.5

S = 4e-2

kd = 5
ke = 1

phi_in = 0
phi_c = 0.6

def rhs(t, u):
    phi = u[:nx]
    h = u[nx:]
    u = 1/(1 - h)
    transfer = kd*S*phi - ke*np.heaviside(h, 0)*(np.maximum(u**2 - u_crit**2, 0))**(3/2)
    dh_dt = transfer/phi_c
    dphi_dt = 1/(1-h[1:])*(dh_dt[1:]*phi[1:] - (phi[1:] - phi[:-1])/dx - transfer[1:])
    return np.concatenate((np.array([0]), dphi_dt, dh_dt))

from scipy.integrate import solve_ivp

h0 = 0.8*np.exp(-100*(np.linspace(0, 1, nx)-0.5)**2)
u0 = np.concatenate((np.array([phi_in]), np.zeros(nx-1), h0))

t_end = 0.8
n_t = 500

sol = solve_ivp(rhs, (0, t_end), u0, method='RK45', rtol=1e-6, t_eval=np.linspace(0, t_end, num=n_t))

import matplotlib.pyplot as plt

phi = sol.y[:nx, :]
h = sol.y[nx:, :]


# animation
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), tight_layout=True)
line1, = ax1.plot(np.linspace(0, L, nx), phi[:, 0])
ax1.set_xlabel("x")
ax1.set_ylabel(r"$\phi$")
line2, = ax2.plot(np.linspace(0, L, nx), h[:, 0])
ax1.set_ylim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_xlabel("x")
ax2.set_ylabel("h")

def update(frame):
    line1.set_ydata(phi[:, frame])
    line2.set_ydata(h[:, frame])

ani = animation.FuncAnimation(fig, update, frames=sol.t.size, interval=1)
# ani.save("steady.mp4", writer='ffmpeg', fps=30)
plt.show()

# static plot for the report
frames = np.linspace(0, n_t-1, 30, dtype=int)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), tight_layout=True)
ax1.plot(np.linspace(0, L, nx), phi[:, frames], 'k')
ax2.plot(np.linspace(0, L, nx), h[:, frames], 'k')
ax1.set_xlabel("x")
ax1.set_ylabel(r"$\phi$")
ax1.set_ylim(0, 1)
ax1.set_xlim(0, 1)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_xlabel("x")
ax2.set_ylabel("h")
plt.savefig("1d_lump.pdf")
plt.show()