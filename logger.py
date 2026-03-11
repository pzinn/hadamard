if __name__ == "__main__":
    raise SystemExit("please run hadamard.py")

import params
from params import config, n
import math
import glob
import re
import datetime
import os

if params.logging == 'wandb':
    import wandb
    wandb_entity = 'aiformath'
    wandb_project = 'topsekrit'


# helper function
def find_latest_gen():
    # Get all filenames matching the pattern
    files = glob.glob(params.work_dir+"GEN-*.txt")
    # Extract the numerical part using regex
    indices = [int(re.search(r"GEN-(\d+)\.txt", f).group(1)) for f in files if re.search(r"GEN-(\d+)\.txt", f)]
    return max(indices) if indices else 0  # Return max index, or 0 if no files found


def init_logging():
    global record_loss, record_scores
    global stats_file, hada_file

    # directory, gen
    if params.resume:
        # existing directory, default is latest
        if not hasattr(params, "work_dir"):
            params.work_dir = os.readlink("latest")
        if not params.work_dir.endswith('/'):
            params.work_dir += '/'  # add trailing /
        # initialise gen if necessary
        if not hasattr(params, "gen"):
            params.gen = find_latest_gen()
    else:
        # make directory
        date = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
        while True:
            params.work_dir = f'training/{n}/{date}/'
            try:
                os.makedirs(params.work_dir, exist_ok=False)
                break
            except FileExistsError:
                date += "_"
        #
        params.gen = 0

    try:
        os.unlink("latest")
    except FileNotFoundError:
        pass
    except IsADirectoryError:
        os.rmdir("latest")
    os.symlink(params.work_dir, "latest")

    if params.logging == 'wandb':
        if params.logging_mode == 'offline':
            os.environ["WANDB_MODE"] = "offline"
        if params.resume:
            with open(params.work_dir + "wandb_run_id.txt") as f:
                run_id = f.read().strip()
            run = wandb.init(entity=wandb_entity, project=wandb_project, id=run_id, dir=params.work_dir,
                             config=config, # is_sweep and resume are incompatible
                             resume='allow',
                             mode = params.logging_mode)
        else:
            myname = f'{params.n}_{date}'
            run = wandb.init(entity=wandb_entity, project=wandb_project, name=myname, dir=params.work_dir,
                             config=config if not params.is_sweep else None,  # if is_sweep, config will be determined dynamically
                             resume='never',
                             mode = params.logging_mode)
            with open(params.work_dir + "wandb_run_id.txt", "w") as f:
                f.write(run.id)
        config.update()  # for sweep
        norm = 1/(math.log(2)*config.stacking)  # renormalise loss so it starts at 1
        def record_loss(loss, step, name):
            wandb.log({"step": step, "loss/"+name+"/"+str(params.gen): norm*loss})
            print(f"{name} {loss=:.6f}", end='\t')
        def record_scores(prefix, scores, mean_score, gens_tally, nh):
            table = wandb.Table(columns=["gen", "count"])
            for g, c in gens_tally.items():
                table.add_data(g, c)
            wandb.log({"gen": params.gen, "score/"+prefix: mean_score, "zero score/"+prefix: nh,
                       "histogram/scores/"+prefix: wandb.Histogram(scores), "table/gens/"+prefix: table})

    # header of stats file
    stats_file = params.work_dir + 'stats.txt'  # where to save logs
    hada_file = params.work_dir + 'hada.txt'  # where to save Hadamard matrices
    with open(stats_file, 'a') as file:
        file.writelines(f"{name}={value!r}\n" for name, value in vars(config).items())

    if params.logging == 'tensorboard':
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=params.work_dir)
        layout = {"combined": {"loss": ["Multiline", ["Loss/train", "Loss/test"]],
                               "score": ["Multline", ["Score/sample", "Score/improved", "Score/selected"]],
                               "zero_score": ["Multline", ["Zero_score/sample", "Zero_score/improved", "Zero_score/selected"]],
                               }
                  }
        writer.add_custom_scalars(layout)
        def record_loss(loss, step, name):
            writer.add_scalar("Loss/"+name, norm*loss, step)
            writer.flush()
            print(f"{name} {loss=:.6f}", end='\t')
        norm = 1/(math.log(2)*config.stacking)  # renormalise loss so it starts at 1
        def record_scores(prefix, scores, mean_score, gens_tally, nh):
            writer.add_scalar("Score/"+prefix, mean_score, params.gen)
            writer.add_scalar("Zero_score/"+prefix, nh, params.gen)
            writer.flush()

    if params.logging == '':  # useful for testing/debugging
        def record_loss(loss, step, name):
            print(f"{name} {loss=:.6f}", end='\t')
        def record_scores(prefix, scores, mean_score, gens_tally, nh):
            pass


if params.is_sweep:
    assert params.logging == 'wandb' and params.logging_mode == 'online', 'sweep only possible with wandb in online mode, i.e. with internet connection to wandb servers.'
    sweep_id = wandb.sweep(entity=wandb_entity, project=wandb_project, sweep=params.sweep_config)
