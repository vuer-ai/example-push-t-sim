import lucidxr_base.train.convert_lucidxr_dataset_hd5 as convert


if __name__ == "__main__":
    convert.main(
        dataset_directory="/lucidxr/lucidxr/datasets/lucidxr/datasets/pnp/data/2025/01/17/dataset1/",
        camera_names=["main"],
        vision_key="render_depth",
    )


