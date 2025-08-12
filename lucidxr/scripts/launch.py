from params_proto import Flag, Proto


def main():
    import os

    import jaynes
    from params_proto import ParamsProto
    from params_proto.hyper import Sweep

    from lucidxr.learning.act_config import ACT_Config

    # make sure lucidxr is loaded first.
    from lucidxr.learning.train import TrainArgs, main
    from lucidxr_experiments import RUN, instr

    class Args(ParamsProto):
        name: str = "pick_block"
        sweep = Proto(help="the jsonl file containing the sweep.")
        local = Flag("run the job locally.")
        sweep = "../../lucidxr_experiments/corl_2025/supplementary/yajvan/mug_tree_new.jsonl"
        chain_length = 1
        # local = True

    sweep = Sweep(RUN, TrainArgs, ACT_Config).load(Args.sweep)

    if RUN.debug:
        print("Setting CUDA_LAUNCH_BLOCKING=1")
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    if Args.local:
        jaynes.config("local")
    else:
        jaynes.config(mode="train")

    print("Number of jobs:", len(sweep))


    prev_job = None
    count = 0
    for i, deps in sweep.items():
        print("job", i)
        if Args.local:
            jaynes.config("local", runner=dict(name=deps["RUN.job_name"]))
        else:
            jaynes.config(mode="train", runner=dict(name=deps["RUN.job_name"]))
        thunk = instr(main)
        if Args.local:
            jaynes.run(thunk, deps)
        else:
            if count == 0:
                prev_job = jaynes.add(thunk, deps)
            else:
                prev_job.chain(thunk, deps)
            count += 1
            if count == Args.chain_length:
                count = 0
                prev_job = None

    jaynes.execute()
    jaynes.listen()


if __name__ == "__main__":
    main()
