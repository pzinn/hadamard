Train a transformer to produce Hadamard matrices.
Modify the parameters in `params.py` (size of matrix, transformer parameters, etc) then run `hadamard.py`
Data are saved in subdirectories of `training/`; runs can be resumed with the parameter `resume` in `params.py`.
A symlink `latest/` points to the latest data.
Hadamard matrices for each run are saved in `hada.txt`.
Use `check.py` to check that they are indeed hadamard matrices, e.g.,
```python
python check.py < latest/hada.txt | uniq
```
