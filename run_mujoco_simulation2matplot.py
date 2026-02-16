import mujoco
import mujoco.viewer
from so101_mujoco_utils2 import set_initial_pose
import numpy as np
import matplotlib.pyplot as plt

MODEL_PATH = "model/scene_urdf.xml"
m = mujoco.MjModel.from_xml_path(MODEL_PATH)
d = mujoco.MjData(m)

# Joint names
joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]

# Starting and target positions (degrees)
starting_position = {
    "shoulder_pan":  -4.4003,
    "shoulder_lift": -92.2462,
    "elbow_flex":     89.9543,
    "wrist_flex":     55.1185,
    "wrist_roll":      0.0,
    "gripper":         0.0,
}

desired_zero = {jn: 0.0 for jn in joint_names}

# Convert degrees to radians
def deg2rad(dct):
    return {k: (v * np.pi/180 if k != "gripper" else v/100*np.pi)
            for k,v in dct.items()}

starting_qpos = deg2rad(starting_position)
target_qpos = deg2rad(desired_zero)

# PID gains
Kp = 50
Kd = 5
Ki = 0.5

integral_error = {jn: 0.0 for jn in joint_names}

# Disturbance parameters
disturbance_time = 0.5
disturbance_duration = 0.2
disturbance_torque = 6.0   # Nm
disturbed_joint = "elbow_flex"

# Data for plotting
data = {jn: [] for jn in joint_names}
time_list = []

# Apply starting pose
set_initial_pose(m, d, starting_position)

with mujoco.viewer.launch_passive(m, d) as viewer:

    max_steps_per_goal = 800
    trajectory = [target_qpos]

    for goal_qpos in trajectory:
        step_count = 0

        while step_count < max_steps_per_goal:

            qpos = {jn: d.qpos[i] for i, jn in enumerate(joint_names)}
            qvel = {jn: d.qvel[i] for i, jn in enumerate(joint_names)}

            done = True

            for i, jn in enumerate(joint_names):
                error = goal_qpos[jn] - qpos[jn]
                integral_error[jn] += error * m.opt.timestep
                derivative = -qvel[jn]

                torque = Kp*error + Ki*integral_error[jn] + Kd*derivative
                d.ctrl[i] = torque

                if abs(error) > 0.01:
                    done = False

            # 🔴 APPLY DISTURBANCE
            if disturbance_time <= d.time <= disturbance_time + disturbance_duration:
                joint_index = joint_names.index(disturbed_joint)
                d.qfrc_applied[joint_index] = disturbance_torque
            else:
                d.qfrc_applied[:] = 0.0

            mujoco.mj_step(m, d)
            viewer.sync()

            time_list.append(float(d.time))
            for jn in joint_names:
                idx = joint_names.index(jn)
                data[jn].append(d.qpos[idx] * 180/np.pi)

            step_count += 1

            pass


# Plot
plt.figure(figsize=(12,6))
for jn in joint_names:
    plt.plot(time_list, data[jn], label=jn)

plt.axvline(x=disturbance_time, color='r', linestyle='--', label='Disturbance')

plt.xlabel("Time (s)")
plt.ylabel("Joint Position (deg)")
plt.title("Joint Positions with External Disturbance")
plt.legend()
plt.grid(True)
plt.show()
