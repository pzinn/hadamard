if __name__ == "__main__":
    raise SystemExit("please run hadamard.py")

import params
import math
import subprocess
from params import n, sample_size, training_size, learning_rate, max_iterations, training_steps, training_batch_size, score_function, num_improve, random_seed

wandb_entity = 'aiformath'
wandb_project = 'topsekrit'

def init_logging():
    global record_loss, record_scores  # ugly TODO better
    global stats_file, hada_file
    global version

    # header of stats file
    stats_file = params.work_dir + 'stats.txt'  # where to save logs
    hada_file = params.work_dir + 'hada.txt'  # where to save Hadamard matrices
    version = subprocess.check_output(["git", "show", "-s", "--pretty='%D %h'"]).strip().decode()
    hparam_list = ['n', 'sample_size', 'training_size', 'learning_rate', 'max_iterations', 'training_steps', 'training_batch_size', 'score_function', 'num_improve', 'version', 'random_seed']  # exclude config because of sweeps

    if params.logging == 'tensorboard':
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=params.work_dir)
        layout = {"combined": {"loss": ["Multiline", ["Loss/train", "Loss/test"]],
                               "score": ["Multline", ["Score/sample", "Score/improved", "Score/selected"]],
                               "zero_score": ["Multline", ["Zero_score/sample", "Zero_score/improved", "Zero_score/selected"]],
                               }
                  }
        writer.add_custom_scalars(layout)
        with open(stats_file, 'a') as file:
            file.writelines(f"{name}={globals().get(name)!r}\n" for name in hparam_list)
            file.write(f"config={params.config}\n")
        def record_loss(loss, step, name):
            writer.add_scalar("Loss/"+name, norm*loss, step)
            writer.flush()
            print(f"{name} {loss=:.6f}", end='\t')
        norm = 1/(math.log(2)*params.config.stacking)  # renormalise loss so it starts at 1
        def record_scores(prefix, scores, gens, mean_score, nh):
            writer.add_scalar("Score/"+prefix, mean_score, params.gen)
            writer.add_scalar("Zero_score/"+prefix, nh, params.gen)
            writer.flush()
    elif params.logging == 'wandb':
        import wandb
        import datetime
        date = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
        myname = f'{params.n}_{date}_{params.sample_size}_{params.training_size}'
        fixed_config = {name: globals().get(name) for name in hparam_list}
        if not params.is_sweep:
            fixed_config.update(params.transformer_config)
        wandb.init(entity=wandb_entity, project=wandb_project, name=myname, id=myname, config=fixed_config, resume=params.resume)  # if sweep mode, resume not supported -- also config is not up to date yet
        with open(stats_file, 'a') as file:
            file.writelines(f"{name}={value!r}\n" for name, value in wandb.config.items())
        norm = 1/(math.log(2)*wandb.config.stacking)  # renormalise loss so it starts at 1
        def record_loss(loss, step, name):
            wandb.log({"step": step, "loss/"+name+"/"+str(params.gen): norm*loss}, commit=name == 'test')  # hacky
            print(f"{name} {loss=:.6f}", end='\t')
        def record_scores(prefix, scores, gens, mean_score, nh):
            wandb.log({"gen": params.gen, "score/"+prefix: mean_score, "zero score/"+prefix: nh,
                       "histogram/scores/"+prefix: wandb.Histogram(scores), "histogram/gens/"+prefix: wandb.Histogram(gens)})


if params.is_sweep:
    import wandb  # compulsory
    sweep_id = wandb.sweep(entity=wandb_entity, project=wandb_project, sweep=params.sweep_config)
