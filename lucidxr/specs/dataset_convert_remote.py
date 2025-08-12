from lucidxr.train.convert_lucidxr_dataset_hd5_remote import main

if __name__ == "__main__":
    main(
        dataset_directory="/lucidxr/lucidxr/datasets/lucidxr/poc/demos/pnp/data/universal_robots_ur5e/2025/01/18/23.48.11",
        camera_names=["table_pov", "overhead"],
        vision_key="render_depth",
        env_id="ur5_basic-pnp-depth-history-v1",
    )
