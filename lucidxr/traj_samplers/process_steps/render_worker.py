import importlib
import json
import os

from dotvar import auto_load  # noqa
import random
import warnings
from collections import defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile

from lucidxr.learning.playback_policy import PlaybackPolicy
from vuer_mujoco.tasks.base.lucidxr_task import get_body_id
from vuer_mujoco.wrappers.utils.tf_utils import get_camera_intrinsic_matrix
import h5py
import numpy as np
import pandas as pd
from ml_logger import logger
from params_proto import Flag, Proto, PrefixProto
from termcolor import colored
from tqdm import tqdm
from zaku import TaskQ
import time

from vuer_mujoco.tasks import make
from vuer_mujoco.wrappers.camera_wrapper import get_image_keys


class RenderParams(PrefixProto, cli_parse=False):
    """Script for collecting virtual reality demos.

    - [x] install collect_demo.py as a cli
    - [x] write params-proto for setting the work dir etc.
    - [x] load an example scene (UR5)
    - [x] ask Yajjy to make a scene with a UR5 and a table
          Ge ended up doing it himself.
    - [x] add ml-logger prefix/dir structure.
    - [x] document in the docs / Notion page.
    """

    wd: str = Proto(env="PWD", help="Working directory")
    vuer_port = Proto(8012, env="PORT", help="vuer port")

    name: str = "scene"
    assets: str = "{name}"
    entry_file: str = "{name}.mjcf.xml"
    camera_keys: list = None

    env_name = "{name}-lucid-v1"
    demo_prefix: str = "lucidxr/lucidxr/datasets/lucidxr/rss-demos/{name}/2025/04/11/16.18.09"
    all_files = Flag(
        "Process all files in the directory",
    )
    overwrite = Flag("Overwrite the existing files")

    playback_policy = Flag("Playback the demo in the vuer instead of setting the qpos")

    start_frame = 0
    end_frame = None

    show_images = Flag("Show images in a matplotlib window")
    no_video = Flag("Save videos to disk")

    dry_run = Flag("Dry run")
    verbose = Flag(help="Print out the assets that are being loaded.")

    lucid_mode = Flag("Use lucid mode")

    def __post_init__(self):
        for k, v in self.__dict__.items():
            if isinstance(v, str):
                value = v.format(**self.__dict__)
                setattr(self, k, value)

                if self.verbose:
                    print(f"{colored(k, 'cyan')}:\t{colored(value, 'yellow')}")



class LucidParams(PrefixProto, cli_parse=False):
    weaver_queue_name: str = Proto(env="$ZAKU_USER:lucidxr:weaver-queue-1")

    crop_size = (1280, 720)  # NOTE THIS IS PIL SIZE (WIDTH, HEIGHT) not npy size (HEIGHT, WIDTH)
    downscale = 2

    camera_keys = ["left_stereo_left"]
    object_key = "ball"
    prompt_jsonl_file: str = Proto("mug_tree.jsonl", help="JSONL file with prompts for the Imagen workflow")

    idle_counter: int = 0
    workflow_cls = "weaver.workflows.lucidxr_workflow:Imagen"
    workflow_arg_keys = {
        "depth": "depth",
        "segmentation": "rgb",
        "object_mask": "ball",
    }
    consistent_multiview = True

    def __post_init__(self, worker_kwargs=None):
        package_name = "vuer_mujoco"
        spec = importlib.util.find_spec(package_name)
        tasks_root = os.path.join(os.path.dirname(spec.origin), "tasks")
        with open(os.path.join(tasks_root, self.prompt_jsonl_file), "r") as file:
            self.prompts = [json.loads(line) for line in file]
        self.default_image = np.zeros((self.crop_size[1] // self.downscale, self.crop_size[0] // self.downscale, 3), dtype=np.uint8)
        self.weaver_queue = TaskQ(name=self.weaver_queue_name, ttl=15)


def show_image(img, obs_key="Observation"):
    import matplotlib.pyplot as plt

    plt.imshow(img)
    # plt.colorbar()
    plt.title(obs_key)
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.show()


def make_videos(image_keys, episode_stem, frames, loader):
    for obs_key in tqdm(image_keys, desc="Making videos"):
        img_prefix = f"render/{episode_stem}/{obs_key}/"
        video_path = f"videos/{episode_stem}/{obs_key}.mp4"
        if loader.glob(f"{video_path}") and len(loader.glob(f"{img_prefix}/*.png")) == 0:
            continue

        counter = 0
        while len(loader.glob(f"{img_prefix}/*.png")) != len(frames):
            counter += 1
            print(f"Waiting for images in {img_prefix} to be ready...")
            print(f"Found {len(loader.glob(f'{img_prefix}/*.png'))} images, expected {len(frames)}")
            time.sleep(1.0)
            if counter > 240:
                # raise ValueError(f"Images in {img_prefix} are not ready after 120 seconds.")
                warnings.warn(f"Images in {img_prefix} are not ready after 120 seconds.")
                counter = 0

        loader.make_video(f"{img_prefix}/*.png", video_path, fps=30, macro_block_size=None)
        # loader.remove(f"{img_prefix}/*.png")
        print("Saved video to", video_path)


def save_h5(episode_stem, data_cache, loader):
    with NamedTemporaryFile(suffix=".h5", delete=False) as f:
        with h5py.File(f.name, "w") as handle:
            for k, d in data_cache.items():
                stacked = np.stack(d)
                print("\033[91m", k, stacked[:, -1].min(), stacked[:, -1].max(), "\033[0m")
                dataset_name = f"{k}"
                handle.create_dataset(dataset_name, data=stacked)
                print(f"Created dataset {dataset_name} with shape {stacked.shape}")

        path = f"data/{episode_stem}.h5"
        loader.upload_file(f.name, path)
        print("Saved rollout to \033[92m", path, "\033[0m")


def add_weaver_job(render_args, lucid_args, **job_kwargs):
    lucid_args.camera_keys = job_kwargs.get("image_keys", lucid_args.camera_keys)
    prompt = random.choice(lucid_args.prompts)
    for key in lucid_args.camera_keys:
        if not key.endswith("lucid"):
            continue
        img_prefix = f"render/{job_kwargs['episode_stem']}/{key}/"
        generated_image_path = img_prefix + f"{job_kwargs['frame_id']:05d}.png"

        workflow_args = {
            k: job_kwargs.get(f"{key}/{v}", lucid_args.default_image) if "image" in k else v
            for k, v in lucid_args.workflow_arg_keys.items()
        }
        overlay_img = job_kwargs[f"{key}/overlay"]
        overlay_mask = job_kwargs[f"{key}/overlay/mask"]

        if not lucid_args.consistent_multiview:
            prompt = random.choice(lucid_args.prompts)

        try:
            lucid_args.weaver_queue.add(
                value=dict(
                    **prompt,
                    **workflow_args,
                    to_logger=generated_image_path,
                    logger_prefix=render_args.demo_prefix,
                    overlay=overlay_img,
                    overlay_mask=overlay_mask,
                    render_kwargs={
                        "downscale": lucid_args.downscale,
                        "crop_size": lucid_args.crop_size,
                        "workflow_cls": lucid_args.workflow_cls,
                    },
                )
            )
            print("Added to weaver queue...")
        except Exception as e:
            print(colored(f"Error: {e}", "red"))
            result = None


def render_worker(*, pin_z, ep_ind, **deps):
    from ml_logger import ML_Logger
    from params_proto import ARGS
    from vuer import Vuer

    ARGS.parse_args()

    RenderParams._update(**deps)
    Vuer._update(**deps)
    LucidParams._update(**deps)

    args = RenderParams()
    lucid_args = LucidParams()

    print(vars(args))

    loader = ML_Logger(prefix=args.demo_prefix)
    print(loader.glob("*"))
    print(loader.prefix)
    loader.job_started(Params=vars(args))

    print(loader.get_dash_url())

    files = loader.glob(f"**/frames/ep_{ep_ind:05d}.pkl")

    if len(files) == 0:
        raise ValueError(f"No files found in the directory: {args.demo_prefix}")

    session_file_path = random.sample(files, k=1)[0]

    print(f"selected \033[91m{session_file_path}\033[0m")
    df = pd.DataFrame(loader.load_pkl(session_file_path)[0])

    session_prefix = Path(session_file_path).parent.parent
    episode_stem = Path(session_file_path).stem

    env = make(args.env_name, strict=False)

    image_keys = RenderParams.camera_keys or get_image_keys(env)

    # logging intrinsics for warping purposes
    cam_keys = set([k.split("/")[0] for k in image_keys])

    try:
        core_cols = ["mocap_pos", "mocap_quat", "qpos", "qvel", "ctrl"]
        cam_cols = [c for c in image_keys if c in df.columns]
        frames = df[core_cols + cam_cols].dropna()
    except KeyError:
        tqdm.write(f"The data is incomplete in {session_file_path}, skipping")
        return

    if args.playback_policy:
        frames["qpos"] = frames["qpos"].apply(lambda x: x[:14])
        frames["qvel"] = frames["qvel"].apply(lambda x: x[:12])
        # TODO a more portable solution

    print("total frames:", len(frames))

    pbar = tqdm(frames.to_dict(orient="records"), desc=f"Processing \033[91m{episode_stem}\033[0m")
    obs_keys = None

    if args.playback_policy:
        args.playback_policy = PlaybackPolicy(f"{args.demo_prefix}/{session_file_path}", verbose=args.verbose)

    new_image_keys = []
    for obs_key in tqdm(image_keys, desc="Checking videos"):
        video_path = f"videos/{episode_stem}/{obs_key}.mp4"
        if not args.overwrite and loader.glob(f"{video_path}"):
            continue
        new_image_keys.append(obs_key)
    image_keys = new_image_keys

    if not image_keys:
        print(f"No new images to render for {session_file_path}, exiting.")
        return

    env.reset()
    with loader.Prefix(session_prefix):
        data_cache = defaultdict(list)
        new_image_keys = []
        for obs_key in image_keys:
            img_prefix = f"render/{episode_stem}/{obs_key}/"
            if not args.overwrite and len(loader.glob(f"{img_prefix}/*.png")) == len(frames):
                logger.print(f"Images for {obs_key} already exist, skipping.")
                continue
            new_image_keys.append(obs_key)
            print("Expected images for", obs_key, ":", len(frames), "found:", len(loader.glob(f"{img_prefix}/*.png")))

        if len(new_image_keys) > 0:
            for step, frame in enumerate(pbar):
                frame_id = step + args.start_frame
                if pin_z:
                    frame["mocap_pos"][2] = 0.63 # pin z

                if args.playback_policy and step > 0:
                    action = args.playback_policy(**frame).flatten()
                    obs, reward, done, info = env.step(prev_act)
                    prev_act = action
                else:
                    obs, reward, done, info = env.get_ordi(**frame)
                    prev_act = env.get_prev_action(**frame)

                if obs_keys is None:
                    obs_keys = list(obs.keys())
                    pbar.write(f"Observation keys: {', '.join(obs_keys)}")

                data_cache["state"].append(obs["state"])
                data_cache["action"].append(prev_act)

                for k in cam_keys:
                    data_cache[f"{k}/C2W"].append(obs.get(f"{k}/C2W"))
                    data_cache[f"{k}/K"].append(obs.get(f"{k}/K"))

                if args.end_frame is not None and frame_id >= args.end_frame:
                    pbar.write(f"Reached end frame {args.end_frame}, stopping.")
                    break

                for key in obs_keys:
                    if any([key.startswith(cam_key) for cam_key in new_image_keys]):
                        continue
                    del obs[key]

                # check if the images all exist
                image_keys_to_generate = []
                for obs_key in new_image_keys:
                    img_prefix = f"render/{episode_stem}/{obs_key}/"

                    img_path = img_prefix + f"{frame_id:05d}.png"

                    if not args.overwrite and loader.glob(img_path):
                        if args.verbose:
                            pbar.write(f"image exists: {img_path}")
                        for key in obs_keys:
                            if key.startswith(obs_key) and key in obs:
                                del obs[key]
                        continue
                    image_keys_to_generate.append(obs_key)
                    img = obs.get(obs_key, None)

                    if img is None:
                        print(f"\033[91mWarning: {obs_key} not found in the observation, skipping\033[0m")
                        continue

                    if not args.dry_run:
                        loader.save_image(img, img_path)
                        if args.verbose:
                            print("Saved image to", img_path)

                    if args.show_images:
                        show_image(img, obs_key=obs_key)


                if args.lucid_mode:
                    conditioning_kwargs = {k: obs[k] for k in obs if "lucid" in k and "raw_segmentation" not in k}
                    if conditioning_kwargs:
                        for key in image_keys_to_generate:
                            print("Adding to weaver queue...")
                            add_weaver_job(
                                render_args=args,
                                lucid_args=lucid_args,
                                **{
                                    "episode_stem": episode_stem,
                                    "frame_id": frame_id,
                                    **conditioning_kwargs,
                                },
                            )

        if not args.no_video:
            make_videos(
                image_keys=image_keys,
                episode_stem=episode_stem,
                frames=frames,
                loader=loader,
            )

        save_h5(
            episode_stem=episode_stem,
            data_cache=data_cache,
            loader=loader,
        )


class MujocoRenderNode:
    """A node that renders the images from the environment."""

    def __init__(self, queue_name, verbose=False, pin_z = False):
        self.queue_name = queue_name
        self.ep_queue = TaskQ(name=self.queue_name, ttl=15)
        self.verbose = verbose
        self.idle_counter = 0
        self.pin_z = pin_z

    def run(self):
        import time

        print("Starting MujocoRenderNode with queue:", self.queue_name)
        while True:
            with self.ep_queue.pop() as job_kwargs:
                if job_kwargs is None:
                    if self.verbose:
                        print(".", end="")
                    time.sleep(1.0)
                    self.idle_counter += 1
                    if self.idle_counter > 60:
                        print("No jobs in the queue for a while, killing node...")
                        return
                    continue
                self.idle_counter = 0
                # print(f"kwargs: {job_kwargs}")

                print(f"Processing job with ep_ind: {job_kwargs.get('ep_ind', 'unknown')}")
                try:
                    render_worker(pin_z = self.pin_z, **job_kwargs)
                except Exception as e:
                    import traceback

                    logger.print(
                        f"Caught Error: {e}. Retrying...",
                        color="red",
                    )
                    traceback.print_exc()
                    self.ep_queue.add(job_kwargs)
                    import random

                    time.sleep(random.uniform(1, 3))


def mujoco_render_entrypoint(
    queue_name,
    pin_z = False,
):
    node = MujocoRenderNode(queue_name, pin_z)
    node.run()


def alan_pick_place():
    """Test the lucid mode."""
    import jaynes

    jaynes.config(mode="local")
    for ep_idx in range(1, 22):
        jaynes.run(
            render_worker,
            ep_ind=ep_idx,
            name="pick_place",
            start_frame=0,
            demo_prefix="/lucidxr/lucidxr/datasets/lucidxr/corl-2025/pick_place/2025/06/01/00.30.21/",
            end_frame=None,
            camera_keys=["left/rgb", "front/rgb", "wrist/rgb"],
            env_name="PickPlace-fixed-v1",
            lucid_mode=False,
            overwrite=True,
            dry_run=False,
            show_images=False,
        )
    jaynes.execute()


def alan_pick_place_random():
    """Test the lucid mode."""
    import jaynes

    jaynes.config(mode="local")
    for ep_idx in range(13, 53):
        jaynes.run(
            render_worker,
            ep_ind=ep_idx,
            name="pick_place",
            start_frame=0,
            demo_prefix="/lucidxr/lucidxr/datasets/lucidxr/corl-2025/pick_place/2025/06/01/16.16.37/",
            end_frame=None,
            camera_keys=["left/rgb", "front/rgb", "wrist/rgb"],
            env_name="PickPlace-block_rand-v1",
            lucid_mode=False,
            overwrite=True,
            dry_run=False,
            show_images=False,
        )


def adam_test():
    render_worker(
        ep_ind=1,
        name="mug_tree_ur",
        start_frame=0,
        demo_prefix="/lucidxr/lucidxr/datasets/lucidxr/corl-2025/mug_tree_ur/2025/06/17/15.53.21/",
        end_frame=None,
        camera_keys=[],
        generative_image_keys=["right", "wrist", "left"],
        env_name="MugTreeUr-fixed-lucid-v1",
        lucid_mode=True,
        overwrite=True,
        dry_run=False,
        show_images=False,
        generative_workflow_arg_keys={
            "image_0": "tree",
            "image_1": "mug",
            "image_2": "midas_depth",
        },
        generative_workflow_cls="weaver.workflows.lucidxr_2_object_mask_workflow:Imagen",
        prompt_jsonl_file="vuer_mujoco/tasks/mug_tree.jsonl",
    )


def adam_test_dr():
    render_worker(
        ep_ind=1,
        name="mug_tree_ur",
        start_frame=0,
        demo_prefix="/lucidxr/lucidxr/datasets/lucidxr/corl-2025/mug_tree_ur/2025/06/18/15.53.21/",
        end_frame=1,
        camera_keys=["left/rgb", "right/rgb", "wrist/rgb"],
        env_name="MugTreeUr-fixed-camera_rand-v1",
        lucid_mode=False,
        overwrite=True,
        dry_run=False,
        show_images=False,
    )


def reproduce_yajvan():
    # quick run to fix the missing data in h5s.
    for i in range(128, 140):
        render_worker(
            ep_ind=i,
            name="mug_tree",
            start_frame=0,
            demo_prefix="/lucidxr/lucidxr/datasets/lucidxr/corl-2025/mug_tree/2025/06/15/16.46.09/",
            end_frame=None,
            camera_keys=[],
            env_name="MugTree-mug_rand-v1",
            lucid_mode=False,
            overwrite=True,
            dry_run=False,
            show_images=False,
        )


def tie_knot_test():
    for i in range(1, 2):
        render_worker(
            ep_ind=i,
            name="tie_knot",
            env_name="TieKnot-lucid-v1",
            camera_keys=["left/lucid/rope", "left/lucid/midas_depth"],
            lucid_mode=False,
            overwrite=True,
            dry_run=False,
            show_images=False,
            playback_policy=True,
            demo_prefix="/lucidxr/lucidxr/datasets/lucidxr/corl-2025/tie_knot/2025/06/30/13.35.46/",
        )


def mug_tree_ur_test():
    for i in range(1, 2):
        render_worker(
            ep_ind=i,
            name="mug_tree",
            env_name="MugTreeUr-mug_rand-v1",
            camera_keys=["left/rgb", "right/rgb", "wrist/rgb"],
            start_frame=0,
            demo_prefix="/lucidxr/lucidxr/datasets/lucidxr/corl-2025/pick_place_robot_room/2025/07/25/18.08.37/",
            end_frame=None,
            lucid_mode=False,
            overwrite=True,
            dry_run=False,
            show_images=False,
            playback_policy=True,
        )


def pick_place_test():
    render_worker(
        ep_ind=1,
        name="pick_place_robot_room",
        env_name="PickPlaceRobotRoom-fixed-v1",
        camera_keys=["left/rgb", "right/rgb", "wrist/rgb"],
        start_frame=0,
        demo_prefix="/lucidxr/lucidxr/datasets/lucidxr/corl-2025/pick_place_robot_room/2025/07/25/18.08.37/",
        end_frame=None,
        lucid_mode=False,
        overwrite=True,
        dry_run=False,
        show_images=False,
        playback_policy=True,
    )


def lucid_test():
    config = dict(
        name="mug_tree",
        env_name="MugTree-fixed-lucid-v1",
        camera_keys=["left/lucid", "right/lucid", "wrist/lucid"],
        workflow_arg_keys={
            "image_0": "tree",
            "image_1": "mug",
            "image_2": "midas_depth",
        },
        workflow_cls="weaver.workflows.lucidxr_2_object_mask_workflow:Imagen",
        prompt_jsonl_file="mug_tree.jsonl",
        dry_run=False,
        overwrite=False,
        lucid_mode=True,
    )
    demo_prefixes = [
        "/lucidxr/lucidxr/datasets/lucidxr/corl-2025/mug_tree/2025/07/29/16.11.36/",
    ]
    render_worker(
        ep_ind=1,
        demo_prefix=demo_prefixes[0].format(**config),
        **config,
    )

def gsplat_test():
    render_worker(
        ep_ind=1,
        name="mug_tree",
        env_name="MugTree-fixed-gsplat-v1",
        camera_keys=["left/rgb", "right/rgb", "wrist/rgb"],
        start_frame=0,
        demo_prefix="/lucidxr/lucidxr/datasets/lucidxr/corl-2025/mug_tree/2025/07/29/16.11.36/",
        end_frame=None,
        lucid_mode=False,
        overwrite=True,
        dry_run=True,
        show_images=True,
    )

# print(os.environ['CUDA_HOME'])
if __name__ == "__main__":
    # mujoco_render_entrypoint(queue_name="yravan:lucidxr:mujoco-queue-2")
    # tie_knot_test()
    # mug_tree_ur_test()
    # reproduce_yajvan()
    # pick_place_test()
    # adam_test_dr()
    # lucid_test()
    gsplat_test()
