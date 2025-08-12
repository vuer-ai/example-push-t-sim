import matplotlib.pyplot as plt
from lucidxr_base import make

if __name__ == "__main__":
    lucidenv = make("ur5_basic-v1", device="cpu", random=0)
    
    lucidenv.reset()
    initial_moca_pose = lucidenv.unwrapped.env.physics.get_mocap()
    frames = []
    for i in range(10):
        initial_moca_pose[0,0] -= 0.1
        initial_moca_pose[0,1] -= 0.1
        initial_moca_pose[0,2] += 0.1
        _, _, _, info = lucidenv.step(initial_moca_pose)
        frames.append(info["render_rgb"])
    for frame in frames:
        plt.imshow(frame)
        plt.show()
    print("Finished Testing Basic Environment")
    
    lucidenv = make("ur5_basic-depth-v1", device="cpu", random=0)
    
    lucidenv.reset()
    initial_moca_pose = lucidenv.unwrapped.env.physics.get_mocap()
    frames = []
    for i in range(10):
        initial_moca_pose[0,0] -= 0.1
        initial_moca_pose[0,1] -= 0.1
        initial_moca_pose[0,2] += 0.1
        _, _, _, info = lucidenv.step(initial_moca_pose)
        frames.append(info["render_depth"])
    for frame in frames:
        plt.imshow(frame)
        plt.show()
    print("Finished Testing Depth Environment")
    


