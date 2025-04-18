if __name__ == "__main__":
    raise SystemExit("please run hadamard.py")

import params
import re
import math


def init_logging():
    global record_loss  # ugly TODO better
    global record_score
    global record_histogram
    norm = 1/(math.log(2)*params.config.stacking)  # renormalise loss so it starts at 1
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
        def record_score(prefix, mean_score, nh, gen):
            writer.add_scalar("Score/"+prefix, mean_score, gen)
            writer.add_scalar("Zero_score/"+prefix, nh, gen)
            writer.flush()
        def record_histogram(prefix, scores, gens):
            pass  # TODO
    elif params.logging == 'wandb':
        import wandb
        myid = re.sub('training', '', re.sub('/', '', params.work_dir))
        wandb.init(entity='pzinn-the-university-of-melbourne', project='sekrit', name=params.work_dir, config=params.config, resume=params.resume, id=myid)
        def record_loss(loss, step, name):
            wandb.log({"step": step, "loss/"+name+"/"+str(params.gen): norm*loss}, commit=name == 'test')  # hacky
            print(f"{name} {loss=:.6f}", end='\t')
        def record_score(prefix, mean_score, nh):
            wandb.log({"gen": params.gen, "score/"+prefix: mean_score, "zero score/"+prefix: nh})
        def record_histogram(prefix, scores, gens):
            # wandb.summary["histogram/scores/"+prefix] = wandb.Histogram(scores)
            # wandb.summary["histogram/gens/"+prefix] = wandb.Histogram(gens)
            wandb.log({"gen": params.gen, "histogram/scores/"+prefix: wandb.Histogram(scores), "histogram/gens/"+prefix: wandb.Histogram(gens)})


if params.is_sweep:
    import wandb  # compulsory
    sweep_id = wandb.sweep(entity='pzinn-the-university-of-melbourne', project="sekrit", sweep=params.sweep_config)
