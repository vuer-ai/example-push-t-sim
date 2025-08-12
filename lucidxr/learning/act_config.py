from params_proto import PrefixProto


class ACT_Config(PrefixProto):
    # ROBOT_STATE_DIM = 7
    # ENV_STATE_DIM = 7

    # input and output dimensions
    action_dim = 9
    obs_dim = 9
    head_dim: int = 128
    vae_latent_dim = 512

    lr: float = 1e-5
    lr_backbone: float = 5e-6
    # batch_size: int = 2 # Not used.
    weight_decay: float = 1e-4
    # epochs: int = 300
    lr_drop: int = 0
    clip_max_norm: float = 0.1
    backbone: str = "resnet18"
    # dilation: bool = False
    position_embedding: str = "sine"
    image_keys: list = ["wrist/rgb", "front/rgb", "left/rgb"]

    enc_layers: int = 4
    dec_layers: int = 7
    dim_feedforward: int = 256
    dropout: float = 0.1
    nheads: int = 8
    chunk_size: int = 10
    pre_norm: bool = False
    action_weighting_factor: float = 0.1

    detr_intermediate_layer = -1 # 0 for first transformer layer, 1 for second, etc.

    normalize_actions: bool = True
    normalize_obs: bool = True

    # masks: bool = False

    eval: bool = False
    onscreen_render: bool = False
    ckpt_dir: str = None
    policy_class: str = None
    task_name: str = None
    seed: int = None
    num_epochs: int = None
    kl_weight: float = 50
    temporal_agg: bool = False
