import threading
import time
import os
from dotvar import auto_load # noqa

from zaku import TaskQ

life_queue_name = "weaver:monitor:life_queue"
life_queue = TaskQ(name=life_queue_name)


def proof_of_life(queue, name):
    """
    A simple function to ensure the script is running.
    """
    queue.add(value={"node_name": name})

def running_on_cluster():
    return "SLURM_JOB_ID" in os.environ

def register_node(name=None):
  """
  Register the monitor node with the system.
  Starts a background thread that runs proof_of_life every 1 second.
  """

  def life_loop():
    nonlocal name
    while True:
      proof_of_life(life_queue, name=str(name) + f"_{os.environ.get('SLURM_JOB_ID', 'local')}-{os.getpid()}")
      time.sleep(30)

  thread = threading.Thread(target=life_loop, daemon=True)
  thread.start()

  print(f"Monitor node '{name}' registered and heartbeat started.")

def monitor_nodes():
    """
    Monitors nodes by processing heartbeat messages.
    Prints alive nodes every 5 seconds.
    """
    registry = {}  # Maps node names to last heartbeat timestamp
    timeout = 60  # seconds
    summary_interval = 10  # seconds
    last_summary = time.time()

    while True:
        with life_queue.pop() as job_kwargs:
            if job_kwargs is None:
                time.sleep(0.1)
            else:
                # Process incoming heartbeat
                node_name = job_kwargs.get("node_name")
                if node_name:
                    registry[node_name] = time.time()
                    # print(f"Heartbeat received from {node_name}")

        # Remove stale nodes
        now = time.time()
        dead_nodes = [name for name, ts in registry.items() if now - ts > timeout]
        for name in dead_nodes:
            print(f"Node {name} is unresponsive. Removing from registry.")
            del registry[name]

        # Print summary periodically
        if now - last_summary > summary_interval:
            print(f"\n[{time.strftime('%H:%M:%S')}] Alive nodes:")
            now = time.time()
            last_summary = now
            for name, ts in registry.items():
              age = now - ts
              print(f"  {name:<30} {age:5.1f}s since last heartbeat")

            # Count node types
            render_count = sum("render" in name for name in registry)
            generative_count = sum("generative" in name for name in registry)
            weaver_count = sum("weaver" in name for name in registry)
            training_count = sum("train" in name for name in registry)
            total = len(registry)

            print(f"\nSummary: total={total} | render={render_count} | generative={generative_count} | weaver={weaver_count} | training={training_count}\n")

if __name__ == "__main__":
  monitor_nodes()