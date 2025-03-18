Train a transformer to produce Hadamard matrices.
Modify the parameters in `params.py` (size of matrix, transformer parameters, etc) then run:
```python
python hadamard.py
```
Data are saved in subdirectories of `training/`; runs can be resumed with the parameter `resume` in `params.py`.
A symlink `latest/` points to the latest data.
Hadamard matrices for each run are saved in `hada.txt`.
Use `check.py` to check that they are indeed hadamard matrices, e.g.,
```python
python check.py < latest/hada.txt | uniq
```
Logging is in two files
* there is a `tfevents` file which can be visualised with `tensorboard`, and summarises three types of information:
   * Loss in the training process.
   * Mean scores of each generation.
   * Proportion of Hadamard matrices in each generation.
* There is a `stats.txt` that contains complementary information, including tallies of the generations at which matrices have been produced.
