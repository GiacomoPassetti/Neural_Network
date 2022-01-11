To install the extension you first need to change from intel to gcc compilers on the cluster:
module switch intel gcc/10.2

Then you need to load python and cuda modules:
module load cuda/11.2
module load python

Then you need to install the extension (this only has to be done once)
python3 install setup.py --user

After that it can be used by importing the rand_gen function from the rand_gen_cuda library. This is shown in example.py