from torchvision.datasets import MNIST as _MNIST
import torch


class MNISTCustom(_MNIST):
    def __init__(self, image_keys, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_keys = image_keys

    def __getitem__(self, idx):
        import time

        # t0 = time.perf_counter()
        img, target = super().__getitem__(idx)

        actions = img.squeeze()
        # obs = torch.zeros(10)

        # camera_views = {k: torch.zeros((3, 360, 640)) for k in self.image_keys}
        # print(f"MNISTCustom __getitem__ took {time.perf_counter() - t0:.4f} seconds")

        return dict(
            # obs=obs,
            actions=actions,
            # target=target,
            # **camera_views,
        )

class MNIST(_MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("MNIST dataset initialized")
        
    def __getitem__(self, idx):
        import time

        # t0 = time.perf_counter()
        img, target = super().__getitem__(idx)
        # print(f"MNIST __getitem__ took {time.perf_counter() - t0:.8f} seconds")
        return img, target


if __name__ == "__main__":
    
    dataset = MNIST(
        root=".",
        train=True,
        download=True,
        transform=None,
        target_transform=None,
    )
    print(f"Dataset size: {len(dataset)}")
    x, y = dataset[0]
    print(f"Image shape: {x.shape}, Label: {y}")
    exit()
    
    import torch
    from torchvision import transforms

    def pad_transform(x):
        # x is a tensor of shape [C, H, W] where H=W=28
        # Pad with zeros to make it 32x32
        return torch.nn.functional.pad(x, (2, 2, 2, 2), mode="constant", value=0)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            pad_transform,
        ]
    )

    dataset = MNISTCustom(
        root=".",
        image_keys=["left/rgb", "right/rgb"],
        train=True,
        download=True,
        transform=transform,
        target_transform=None,
    )

    print(f"Dataset size: {len(dataset)}")

    from matplotlib import pyplot as plt

    plt.imshow(dataset[0]["action"])
    plt.show()
