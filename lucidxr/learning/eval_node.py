import gc
from time import sleep

from params_proto import ParamsProto
from params_proto import Proto

from lucidxr.learning import unroll_eval as unroll

import signal
from contextlib import contextmanager


@contextmanager
def time_limit(seconds: int):
    """Raise `TimeoutError` if the `with` block takes longer than *seconds*."""

    def _handler(signum, frame):
        raise TimeoutError(f"Exceeded time limit of {seconds}s")

    signal.signal(signal.SIGALRM, _handler)  # install handler
    signal.alarm(seconds)  # start the countdown
    try:
        yield
    finally:
        signal.alarm(0)  # always cancel on exit


class EvalNode(ParamsProto, prefix="eval"):
    """Worker for evaluating an agent"""

    queue_name: str = Proto(env="$ZAKU_USER:lucidxr:eval-queue-1")
    timeout = 10000
    idle_counter: int = 0

    def __post_init__(self, worker_kwargs=None):
        from ml_logger import logger
        from zaku import TaskQ

        print("subscribing to the worker queue", self.queue_name)
        self.queue = TaskQ(name=self.queue_name)

        logger.print("created workflow", color="green")

    def run(self):
        from ml_logger import logger

        print("starting the worker...")
        while True:
            with self.queue.pop() as job_kwargs:
                if job_kwargs is None:
                    logger.print(".", end="", color="yellow")
                    sleep(3.0)  # no need to query frequently when queue is empty.
                    self.idle_counter += 1
                    # if self.idle_counter > 200:
                    #     print("No jobs in the queue for a while, killing node...")
                    #     return
                    continue
                self.idle_counter = 0

                # untested
                if job_kwargs.get("$kill", None):
                    exit()

                print(job_kwargs)
                logger.print("Evaluating...", color="green")
                # unroll.main(job_kwargs)
                try:
                    with time_limit(self.timeout):
                        unroll.main(job_kwargs, strict=False)
                except Exception as e:
                    import traceback
                    logger.print(
                        f"Caught Error: {e} with arguments {job_kwargs}",
                        color="red",
                    )
                    traceback.print_exc()
                    self.queue.add(job_kwargs)
                    # sleep a bit
                    import random
                    import time

                    time.sleep(random.uniform(1, 3))

                gc.collect()


def entrypoint(**deps):
    worker = EvalNode(**deps)
    worker.run()
    from ml_logger import logger
    logger.print("Worker finished running", color="green")


if __name__ == "__main__":
    entrypoint()
