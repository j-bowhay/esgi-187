import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

dt = 0.01
n = 1000

u = 1
s = 0.05

H = 1
L = 10

phi_in = 0.2
phi_c = 0.6

ny = 10
y = np.zeros((n, ny))
y[0, :] = np.linspace(0, 1, ny)
x = np.zeros_like(y)
phi = np.ones_like(y)*phi_in

for i in range(1, n):
    # find the characteristics that have reached y=0
    terminated = y[i - 1, :] <= 0

    # update the characteristics that have not terminated
    x[i, ~terminated] = x[i - 1, ~terminated] + u * dt
    y[i, ~terminated] = y[i - 1, ~terminated] - s * dt
    phi[i, :] = phi[i - 1, :] # for now u is constant so the value of phi does not change

fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
ax0.set_ylim(0, H)
ax0.set_xlim(0, L)

char_line = ax0.plot(x[0, :], y[0, :], '.')

def update(frame):
    char_line[0].set_data(x[:frame, :], y[:frame, :])

ani = animation.FuncAnimation(fig, update, frames=np.arange(0, n, 1), interval=50)
plt.show()