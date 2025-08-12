from pathlib import Path

from dotvar import auto_load  # noqa
from params_proto import Proto, Flag, PrefixProto
import os
os.environ["MUJOCO_GL"]="glfw"


class CleanLucidSim(PrefixProto, cli_parse=False):

    demo_prefix: str = "/lucidxr/lucidxr/datasets/lucidxr/rss-demos/{name}/2025/04/11/15.44.01"


    generative_queue_name: str = Proto(env="$ZAKU_USER:lucidxr:weaver-queue-generative")
    weaver_queue_name: str = Proto(env="$ZAKU_USER:lucidxr:weaver-queue-1")


    def run(self):
        from ml_logger import ML_Logger
        logger = ML_Logger(prefix=self.demo_prefix)
        generated_folder = "generated"
        folders = list(logger.glob(f"{generated_folder}/**/rgb"))
        print("Found folders:", folders)

        for f in folders:
            img_prefix = f"{f}"
            video_path = f"generated_videos/{Path(f).relative_to(generated_folder)}.mp4"
            if logger.glob(f"{video_path}") and len(logger.glob(f"{img_prefix}/*.png")) == 0:
                continue
            print("making video", video_path)
            logger.make_video(f"{img_prefix}/*.png", video_path, fps=30)
            logger.remove(f"{img_prefix}/*.png")

        print("Done cleaning up generative renders.")

        # Clean up the generative queue
        from zaku import TaskQ
        generative_queue = TaskQ(name=self.generative_queue_name)
        weaver_queue = TaskQ(name=self.weaver_queue_name)
        generative_queue.clear_queue()
        weaver_queue.clear_queue()




def entrypoint(**deps):
    worker = CleanLucidSim(**deps)
    worker.run()

if __name__ == "__main__":
    entrypoint(
      demo_prefix = "/lucidxr/lucidxr/datasets/lucidxr/rss-demos/mug_tree/2025/04/29/18.46.35/"
    )
