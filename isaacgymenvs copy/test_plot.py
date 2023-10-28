import numpy as np
import matplotlib.pyplot as plt

# Load data
data = np.load('saved_data.npz')
wR = data['wR']
wL = data['wL']
ww = data['ww']

# Define the wheels names
wheels = ["w1", "w2", "w3", "w4"]

# Get the length of data
data_length = ww.shape[2]

# w1
plt.figure()
plt.title("Wheel: w1")
plt.plot(list(range(data_length)), ww[0, 0, :], label="ww value")
plt.plot(list(range(data_length)), [wR[0]] * data_length, label="wR value")
plt.xlabel('Iterations')
plt.ylabel('Value')
plt.legend()
plt.savefig("w1.png")
plt.show()

# w2
plt.figure()
plt.title("Wheel: w2")
plt.plot(list(range(data_length)), ww[0, 1, :], label="ww value")
plt.plot(list(range(data_length)), [wL[0]] * data_length, label="wL value")
plt.xlabel('Iterations')
plt.ylabel('Value')
plt.legend()
plt.savefig("w2.png")
plt.show()

# w3
plt.figure()
plt.title("Wheel: w3")
plt.plot(list(range(data_length)), ww[0, 2, :], label="ww value")
plt.plot(list(range(data_length)), [wR[0]] * data_length, label="wR value")
plt.xlabel('Iterations')
plt.ylabel('Value')
plt.legend()
plt.savefig("w3.png")
plt.show()

# w4
plt.figure()
plt.title("Wheel: w4")
plt.plot(list(range(data_length)), ww[0, 3, :], label="ww value")
plt.plot(list(range(data_length)), [wL[0]] * data_length, label="wL value")
plt.xlabel('Iterations')
plt.ylabel('Value')
plt.legend()
plt.savefig("w4.png")
plt.show()

# Combined plot for all wheels
plt.figure(figsize=(12, 8))
for idx, wheel in enumerate(wheels):
    plt.plot(list(range(data_length)), ww[0, idx, :], label=wheel)
plt.title("Combined Wheel Data")
plt.xlabel('Iterations')
plt.ylabel('Value')
plt.legend()
plt.savefig("combined_wheels.png")
plt.show()
