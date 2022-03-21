CUDA_VISIBLE_DEVICES=0 python maml.py --seed 1 &
CUDA_VISIBLE_DEVICES=1 python maml.py --seed 2 &
CUDA_VISIBLE_DEVICES=0 python maml.py --seed 3 &
CUDA_VISIBLE_DEVICES=1 python maml.py --seed 4 &
CUDA_VISIBLE_DEVICES=1 python maml.py --seed 5 &
