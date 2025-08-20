I'll just use the miniconda I installed for CS 231N.

```
source ~/.bashrccs231n
```

(the path to CS 231N's miniconda is written in this bashrccs231n file)

Create a brand new environment:

```
conda create -n cs224rproject python==3.8
conda activate cs224rproject
```

### Isaac Gym

Download Isaac Gym from either [their official website](https://developer.nvidia.com/isaac-gym). Then:

```
cd /iris/u/duyi/cs231n/isaacgym/python
pip install -e .
```

Thentry to run an example script:

```
cd examples/
python joint_monkey.py
```

I encountered this error:

```
ImportError: libpython3.7m.so.1.0: cannot open shared object file: No such file or directory
```

I solved this by

```
export LD_LIBRARY_PATH=/iris/u/duyi/cs231n/miniconda3/envs/rlgpu/lib/
```

Now when I run the joint_monkey example, a window pops up and I can see funny white people dancing.

### UniDexGrasp

#### policy

This part is learned from `UniDexGrasp/dexgrasp_policy/README.md`.

```
cd /iris/u/duyi/cs231n/
git clone https://github.com/PKU-EPIC/UniDexGrasp.git
cd /iris/u/duyi/cs231n/UniDexGrasp/dexgrasp_policy/
```

I encountered a problem with the line `python_requires=">=3.6.*"` in setup.py. I fixed it by changing this line to `python_requires=">=3.6"`.

I also encountered a problem that my home directory (`/afs/cs.stanford.edu/u/duyi/`) is too small for building the wheels. Thus, I created a tmp folder in the large disk for caching.

```
mkdir -p /iris/u/duyi/cs231n/tmp
TMPDIR=/iris/u/duyi/cs231n/tmp
pip install -e . --cache-dir /iris/u/duyi/cs231n/tmp/pip-cache
```

According to UniDexGrasp's readme (`dexgrasp_policy/README.md`), we also need to install pointnet2_ops:

```
cd /iris/u/duyi/cs231n/UniDexGrasp
git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git
cd Pointnet2_PyTorch/pointnet2_ops_lib/
python setup.py install
```

#### generation

This part is learned from `UniDexGrasp/dexgrasp_generation/README.md`.

```
cd /iris/u/duyi/cs231n/UniDexGrasp/
conda install -y pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```

...(not completed yet, but we'll skip this and focus on the policy part first)

### test isaac gym

I wrote a simple script to test isaac gym and the provided object mesh (hand and ball).

```python
import os
import numpy as np
from isaacgym import gymutil
from isaacgym import gymapi
from math import sqrt

# initialize gym
gym = gymapi.acquire_gym()
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX)
gym.add_ground(sim, gymapi.PlaneParams())  # Add ground

# ==== load assets ====
original_dir = os.getcwd()  # Save the original directory
os.chdir("/iris/u/duyi/cs231n/UniDexGrasp/dexgrasp_policy/assets/urdf/shadow_hand_description/")

ball  = gym.load_asset(sim, "/iris/u/duyi/cs231n/UniDexGrasp/dexgrasp_policy/assets/urdf/objects/", "/ball.urdf")
print('successfully loaded ball')
hand_new = gym.load_asset(sim, "/iris/u/duyi/cs231n/UniDexGrasp/dexgrasp_policy/assets/urdf/shadow_hand_description/", "/shadowhand_new.urdf")
print('successfully loaded hand_new')
hand = gym.load_asset(sim, "/iris/u/duyi/cs231n/UniDexGrasp/dexgrasp_policy/assets/urdf/shadow_hand_description/", "/shadowhand.urdf")
print('successfully loaded hand')
hand_with_fingertips = gym.load_asset(sim, "/iris/u/duyi/cs231n/UniDexGrasp/dexgrasp_policy/assets/urdf/shadow_hand_description/", "/shadowhand_with_fingertips.urdf")
print('successfully loaded hand_with_fingertips')

os.chdir(original_dir)  # Change back to the original directory

# ==== simple visualization ====
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# In Isaac Gym, the coordinate system is:
# x: right (+) / left (−)
# y: up (+) / down (−) ← vertical direction
# z: forward (+) / backward (−)
spacing = 2.0
lower = gymapi.Vec3(-spacing, 0.0, -spacing)
upper = gymapi.Vec3(spacing, spacing, spacing)
env = gym.create_env(sim, lower, upper, 1)

# Set up the camera
cam_pos = gymapi.Vec3(3.0, 2.0, 3.0)
cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# Create instances of the assets
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)  # quaternion
# Quanterion is a mathematical way to represent rotation
# I don't know details, but (0, 0, 0, 1) basically means no rotation

# Add the ball
ball_handle = gym.create_actor(env, ball, pose, "ball", 0, 0)

# Add the hand
hand_handle = gym.create_actor(env, hand, pose, "hand", 0, 0)

# Main simulation loop
while not gym.query_viewer_has_closed(viewer):
    # Step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    
    # Update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    
    # Wait for dt to elapse in real time
    gym.sync_frame_time(sim)

# Clean up
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
```

This code should pop up a window in Isaac Gym with a hand and a ball laying on the ground.

### Download the dataset

The dataset is [here](https://mirrors.pku.edu.cn/dl-release/UniDexGrasp_CVPR2023/). Because we only care about policy training (not grasp generation) for now, I downloaded the 3 files in `dexgrasp_policy/assets/`: `meshdatav3_pc_feat.zip`, `meshdatav3_pc_fps.zip`, and `meshdatav3_scaled.tar.xz`.

Use scp to upload them to the same assets directory on the cluster. Then unzip them. This may take quite a while, so I recommend doing this in tmux.

```
cd /iris/u/duyi/cs231n/UniDexGrasp/dexgrasp_policy/assets
unzip meshdatav3_pc_feat.zip
unzip meshdatav3_pc_fps.zip
tar -xf meshdatav3_scaled.tar.xz
```

### Sample training (of the policy)

Now we can do a sample training with the following command:

```
cd /iris/u/duyi/cs231n/UniDexGrasp/dexgrasp_policy/dexgrasp
bash script/run_train_ppo_state.sh
```

This script will run `train.py`, which will read the config file `cfg/shadow_hand_grasp.yaml`.

> The first time I ran it, I encountered the following error: `AttributeError: module 'torch' has no attribute 'UntypedStorage'`. This is because my PyTorch version was too old to match my PyTorch Lightening. I solved this by `pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1`.

I was then able to run this training script. It runs for 10,000 iterations. The logs are stored in `UniDexGrasp/dexgrasp_policy/dexgrasp/logs/test_seed0`, which include something training statistics (can be opened in Tensorboard) and model checkpoints (every 500 iterations).

```
tensorboard --logdir logs/test_seed0 --port 6006
```

> I encountered an error `TypeError: MessageToJson() got an unexpected keyword argument 'including_default_value_fields'` when I tried to launch tensorboard. This is because my `protobuf` version is too high to match my tensorboard. I solved this by `pip install protobuf==3.20.3`.

After training the policy, I wrote a script to visualize the grasp in Isaac Gym. (The training part uses `--headless`, so it never visualize anything.) The script is in `UniDexGrasp/dexgrasp_policy/dexgrasp/visualize_rl_results.py`.
