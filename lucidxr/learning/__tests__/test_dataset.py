from pathlib import PosixPath

from dotvar import auto_load  # noqa

from lucidxr.learning.episode_datasets import EpisodeDataset, HistoryDataset


def test_gripper_traj():
    for ep_ind in range(2, 20):
        path = f"00.04.43/data/ep_{ep_ind:05d}.h5"

        dataset = EpisodeDataset(
            data_prefix="lucidxr/lucidxr/datasets/lucidxr/rss-demos/pick_block/2025/03/31",
            cache_root=PosixPath("/home/geyang/.cache"),
            episode_paths=[path],
            chunk_size=100,
            image_keys=["wrist/rgb", "front/rgb", "left/rgb"],
            skip_images=False,
            prune_cache=True,
        )

        return
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.subplot(2, 1, 1)
        plt.plot(dataset.actions)
        plt.title(path + " Actions")
        plt.xlabel("Timesteps")
        plt.ylabel("Action Value")
        plt.legend([f"{ind}" for ind in range(dataset.actions.shape[-1])])

        plt.subplot(2, 1, 2)
        plt.plot(dataset.actions[:, -1])
        plt.title("Gripper Control")
        plt.xlabel("Timesteps")
        plt.tight_layout()

        plt.show(dpi=300)


def test_history_dataset():
    for ep_ind in range(3, 4):
        path = f"00.04.43/data/ep_{ep_ind:05d}.h5"

        dataset = HistoryDataset(
            data_prefix="lucidxr/lucidxr/datasets/lucidxr/rss-demos/pick_block/2025/03/31",
            cache_root=PosixPath("/home/geyang/.cache"),
            episode_paths=[path],
            history_len=3,
            chunk_size=100,
            image_keys=["wrist/rgb", "front/rgb", "left/rgb"],
            skip_images=False,
            prune_cache=False,
        )

        for batch in dataset:
            print(batch["obs"].shape, batch["actions"].shape)
            assert batch["obs"].shape == (3, 10)
            break
