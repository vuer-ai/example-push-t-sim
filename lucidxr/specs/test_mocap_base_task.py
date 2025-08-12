import matplotlib.pyplot as plt
from dm_control.rl import control

from vuer_mujoco.tasks.base.lucidxr_task import SimplePhysics
from vuer_mujoco.tasks.base.mocap_task import MocapTask

# TODO set
SCENE_FILE = "/assets/third_party/adam_robots/universal_robots_ur5e/scene.mjcf.xml"

DEFAULT_TIME_LIMIT = 25

PHYSICS_TIMESTEP = 0.005  # in XML
DECIMATION = 4
CONTROL_TIMESTEP = PHYSICS_TIMESTEP * DECIMATION


def test_mocap_physics(physics):
    pass
    initial_mocap_pose = physics.get_mocap()
    frame1 = physics.render()
    assert initial_mocap_pose.shape == (1, 7)
    initial_mocap_pose[0, 0] -= 0.1
    initial_mocap_pose[0, 1] -= 0.1
    initial_mocap_pose[0, 2] += 0.1
    physics.set_mujoco_data(**initial_mocap_pose)
    physics.step()
    frame2 = physics.render()
    plt.imshow(frame1)
    plt.show()
    plt.imshow(frame2)
    plt.show()
    print("Finished Testing Physics")


def main():
    physics = SimplePhysics.from_xml_path(SCENE_FILE)
    test_mocap_physics(physics)
    task = MocapTask()
    env = control.Environment(
        physics,
        task,
        time_limit=DEFAULT_TIME_LIMIT,
        control_timestep=CONTROL_TIMESTEP,
        flat_observation=True,
    )
    initial_mocap_pose = physics.get_mocap()
    new_mocap_pose = initial_mocap_pose
    frames = [physics.render()]
    for i in range(10):
        new_mocap_pose[0, 0] -= 0.1
        new_mocap_pose[0, 1] -= 0.1
        new_mocap_pose[0, 2] += 0.1
        env.step(new_mocap_pose)
        frames.append(physics.render())
    for frame in frames:
        plt.imshow(frame)
        plt.show()
    print("Finished Testing Task")


if __name__ == "__main__":
    main()
