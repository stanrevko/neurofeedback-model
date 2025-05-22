# This script simulates and animates the dynamics of a single Izhikevich neuron model.
# It shows how the neuron's membrane potential (v) and recovery variable (u) evolve over time
# in response to an external input current. The animation visualizes the neuron's firing behavior,
# including spike events when the membrane potential reaches the threshold.
# Blue line: membrane potential (v)
# Orange line: recovery variable (u)
# Red dots: spike occurrences
# The script helps to intuitively understand the spiking and recovery process of the Izhikevich neuron.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Параметри моделі Іжікевича
a, b, c, d = 0.02, 0.2, -65, 8
dt = 0.5  # ms
t_max = 200  # ms
time = np.arange(0, t_max, dt)

# Ініціалізація змінних
v = c * np.ones_like(time)  # потенціал
u = b * v                   # відновлення

# Зовнішній вхідний струм I (імпульс з 50 по 150 ms)
I = np.zeros_like(time)
I[(time >= 50) & (time < 150)] = 10

# Зберігаємо спайки для відображення
spikes = []

# Функція оновлення моделі за крок dt
def step(i):
    global v, u, spikes
    if v[i-1] >= 30:
        v[i-1] = 30  # відобразити спайк як пікове значення
        v[i] = c
        u[i] = u[i-1] + d
        spikes.append(i-1)
    else:
        dv = (0.04 * v[i-1]**2 + 5 * v[i-1] + 140 - u[i-1] + I[i-1]) * dt
        du = a * (b * v[i-1] - u[i-1]) * dt
        v[i] = v[i-1] + dv
        u[i] = u[i-1] + du

# Обчислюємо динаміку
for i in range(1, len(time)):
    step(i)

# Створюємо анімацію
fig, ax = plt.subplots(figsize=(10,6))
ax.set_xlim(0, t_max)
ax.set_ylim(-80, 40)
ax.set_xlabel('Час (ms)')
ax.set_ylabel('Потенціал v (синій), Відновлення u (помаранчевий)')
ax.set_title('Динаміка нейрона Іжікевича')

line_v, = ax.plot([], [], lw=2, color='blue', label='v (потенціал)')
line_u, = ax.plot([], [], lw=2, color='orange', label='u (відновлення)')
spike_dots, = ax.plot([], [], 'ro', label='Спайки')

ax.legend(loc='upper right')

def init():
    line_v.set_data([], [])
    line_u.set_data([], [])
    spike_dots.set_data([], [])
    return line_v, line_u, spike_dots

def update(frame):
    x = time[:frame]
    y_v = v[:frame]
    y_u = u[:frame]
    spikes_x = [time[s] for s in spikes if s < frame]
    spikes_y = [30 for s in spikes if s < frame]
    line_v.set_data(x, y_v)
    line_u.set_data(x, y_u)
    spike_dots.set_data(spikes_x, spikes_y)
    return line_v, line_u, spike_dots

ani = FuncAnimation(fig, update, frames=len(time),
                    init_func=init, blit=True, interval=30)

plt.show()
