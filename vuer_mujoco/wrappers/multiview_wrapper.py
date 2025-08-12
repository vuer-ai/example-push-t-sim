from vuer_mujoco.wrappers.camera_wrapper import CameraWrapper, DepthWrapper, RGBDWrapper


def create_multiview_env(env, **camera_configs) -> CameraWrapper:
    """
    Creates an environment with multiple camera views and attaches them
    to the observation dict using corresponding image_keys.
    """
    for key, config in camera_configs.items():
        if config.get("depth", None):
            env = DepthWrapper(env, image_key=key, **config)
        elif config.get("rgbd", None):
            env = RGBDWrapper(env, image_key=key, **config)
        elif config.get("lucid", None):
            from vuer_mujoco.wrappers.lucid_wrapper import (
                SegmentationWrapper,
                OverlayWrapper,
                MidasDepthWrapper,
                MaskWrapper,
            )
            env = SegmentationWrapper(env, image_key=key, **config)
            env = OverlayWrapper(env, image_key=key, **config)
            env = MidasDepthWrapper(env, image_key=key, **config)
            if config.get("object_keys", None):
                for obj in config["object_keys"]:
                    env = MaskWrapper(env, image_key=key, object_prefix=obj, **config)
            else:
                raise ValueError("object_key must be specified for lucid segmentation.")
        elif config.get("domain_rand", None):
            from vuer_mujoco.wrappers.domain_randomization_wrapper import DomainRandomizationWrapper
            env = DomainRandomizationWrapper(env, image_key=key, **config)
        elif config.get("gsplat", None):
            from vuer_mujoco.wrappers.lucid_wrapper import (
                OverlayWrapper,
            )
            from vuer_mujoco.neverwhere.gsplat_wrapper import SplatRGBWrapper
            env = OverlayWrapper(env, image_key=key, **config)
            env = SplatRGBWrapper(env, image_key=key, **config)
        else:
            env = CameraWrapper(env, image_key=key, **config)

    return env
