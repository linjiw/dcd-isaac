# import sys
# sys.path.append("/home/linjiw/Downloads/isaacgym-jackal/")
import isaacgym
import isaacgymenvs
import torch
import time
import numpy as np

num_envs = 256

envs = isaacgymenvs.make(
	seed=0, 
	task="Jackal", 
	num_envs=num_envs, 
	sim_device="cuda:0",
	rl_device="cuda:0",
	graphics_device_id=0,
    headless=False
)
print("Observation space is", envs.observation_space)
print("Action space is", envs.action_space)
obs = envs.reset()
t1 = time.time()
rewards = 0

xx, yy = [], []
ww = []
for i in range(200):
	actions = 1 * torch.ones((num_envs, 2), device="cuda:0")
	actions[:, 0] = 0.  # rotate with 2\pi
	obs, reward, done, info = envs.step(actions)
	rewards += reward
	print("iter %d: " %(i+1), info["dof_vel"][0], envs.root_states[0][-4:])
	#print(envs.contact_forces[envs.jackal_rigid_body_idx][0, :3], envs.root_states[envs.jackal_actor_idx][0, 2])
	#print(envs.contact_forces[envs.jackal_rigid_body_idx+1][0, :3])
	#print(envs.contact_forces[envs.jackal_rigid_body_idx+2][0, :3])
	#print(envs.contact_forces[envs.jackal_rigid_body_idx+3][0, :3])
	#print(envs.contact_forces[envs.jackal_rigid_body_idx+4][0, :3])
	x, y = envs.root_states[envs.jackal_actor_idx][:, 0].cpu().detach().numpy(), envs.root_states[envs.jackal_actor_idx][:, 1].cpu().detach().numpy()
	xx.append(x)
	yy.append(y)
	ww.append(info["dof_vel"].cpu().detach().numpy())
	# print(obs["obs"][0, :84].detach().cpu())
t2 = time.time()
print(info["dof_vel"].cpu().detach().numpy())
# print("fps: %.4f" %(128 * 100 / (t2-t1)))
# print("reward: %.4f" %(torch.mean(rewards).detach().cpu()))

from matplotlib import pyplot as plt
xx = np.stack(xx, axis=-1)
yy = np.stack(yy, axis=-1)
ww = np.stack(ww, axis=-1)

for ex, ey in zip(xx, yy):	
	plt.plot(ex, ey, alpha=0.2, color="red")
plt.savefig("jackal.png")

plt.show()
# '''
# actions[:, 0] = actions[:, 0] * 2; actions[:, 1] = actions[:, 1] * 3.14
# wR = (2 * actions[:, 0] + actions[:, 1] * 0.37559) / (2 * 0.098)
# wL = (2 * actions[:, 0] - actions[:, 1] * 0.37559) / (2 * 0.098)
# wR = wR.cpu().detach().numpy()
# wL = wL.cpu().detach().numpy()
# # w1
# plt.plot(list(range(100)), ww[0, 0, :])
# plt.plot(list(range(100)), [wR[0]] * 100)
# plt.show()
# # plt.savefig("w1.png")


# # w2
# plt.plot(list(range(100)), ww[0, 1, :])
# plt.plot(list(range(100)), [wL[0]] * 100)
# plt.show()
# plt.savefig("jackal.png")

# # w3
# plt.plot(list(range(100)), ww[0, 2, :])
# plt.plot(list(range(100)), [wR[0]] * 100)
# plt.show()
# plt.savefig("jackal.png")

# # w4
# plt.plot(list(range(100)), ww[0, 3, :])
# plt.plot(list(range(100)), [wL[0]] * 100)
# plt.show()
# plt.savefig("jackal.png")

# '''

# actions[:, 0] = actions[:, 0] * 2; actions[:, 1] = actions[:, 1] * 3.14
# wR = (2 * actions[:, 0] + actions[:, 1] * 0.37559) / (2 * 0.098)
# wL = (2 * actions[:, 0] - actions[:, 1] * 0.37559) / (2 * 0.098)
# wR = wR.cpu().detach().numpy()
# wL = wL.cpu().detach().numpy()

# # Define the wheels names
# wheels = ["w1", "w2", "w3", "w4"]




# # Combined plot setup
# plt.figure(figsize=(10, 6))

# # w1
# plt.figure()
# plt.title("Wheel: w1")
# plt.plot(list(range(100)), ww[0, 0, :])
# plt.plot(list(range(100)), [wR[0]] * 100)
# plt.savefig("w1.png")
# plt.show()

# # w2
# plt.figure()
# plt.title("Wheel: w2")
# plt.plot(list(range(100)), ww[0, 1, :])
# plt.plot(list(range(100)), [wL[0]] * 100)
# plt.savefig("w2.png")
# plt.show()

# # w3
# plt.figure()
# plt.title("Wheel: w3")
# plt.plot(list(range(100)), ww[0, 2, :])
# plt.plot(list(range(100)), [wR[0]] * 100)
# plt.savefig("w3.png")
# plt.show()

# # w4
# plt.figure()
# plt.title("Wheel: w4")
# plt.plot(list(range(100)), ww[0, 3, :])
# plt.plot(list(range(100)), [wL[0]] * 100)
# plt.savefig("w4.png")
# plt.show()

# # Combined plot for all wheels
# plt.figure(figsize=(12, 8))
# for idx, wheel in enumerate(wheels):
#     plt.plot(list(range(100)), ww[0, idx, :], label=wheel)
# plt.title("Combined Wheel Data")
# plt.legend()
# plt.savefig("combined_wheels.png")
# plt.show()


actions[:, 0] = actions[:, 0] * 2; actions[:, 1] = actions[:, 1] * 3.14
wR = (2 * actions[:, 0] + actions[:, 1] * 0.37559) / (2 * 0.098)
wL = (2 * actions[:, 0] - actions[:, 1] * 0.37559) / (2 * 0.098)
wR = wR.cpu().detach().numpy()
wL = wL.cpu().detach().numpy()
np.savez('saved_data.npz', wR=wR, wL=wL, ww=ww)

# # Define the wheels names
# wheels = ["w1", "w2", "w3", "w4"]

# # Combined plot setup
# plt.figure(figsize=(10, 6))

# # Get the length of data
# data_length = ww.shape[2]

# # w1
# plt.figure()
# plt.title("Wheel: w1")
# plt.plot(list(range(data_length)), ww[0, 0, :])
# plt.plot(list(range(data_length)), [wR[0]] * data_length)
# plt.savefig("w1.png")
# plt.show()

# # w2
# plt.figure()
# plt.title("Wheel: w2")
# plt.plot(list(range(data_length)), ww[0, 1, :])
# plt.plot(list(range(data_length)), [wL[0]] * data_length)
# plt.savefig("w2.png")
# plt.show()

# # w3
# plt.figure()
# plt.title("Wheel: w3")
# plt.plot(list(range(data_length)), ww[0, 2, :])
# plt.plot(list(range(data_length)), [wR[0]] * data_length)
# plt.savefig("w3.png")
# plt.show()

# # w4
# plt.figure()
# plt.title("Wheel: w4")
# plt.plot(list(range(data_length)), ww[0, 3, :])
# plt.plot(list(range(data_length)), [wL[0]] * data_length)
# plt.savefig("w4.png")
# plt.show()

# # Combined plot for all wheels
# plt.figure(figsize=(12, 8))
# for idx, wheel in enumerate(wheels):
#     plt.plot(list(range(data_length)), ww[0, idx, :], label=wheel)
# plt.title("Combined Wheel Data")
# plt.legend()
# plt.savefig("combined_wheels.png")
# plt.show()

# print(f"actions: {actions[0, :]}")
