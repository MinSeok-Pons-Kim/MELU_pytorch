CUDA_VISIBLE_DEVICES=1 python maml.py --seed 1 --test --test_way old_user &
#CUDA_VISIBLE_DEVICES=1 python maml.py --seed 1 --test --test_way new_user &

CUDA_VISIBLE_DEVICES=1 python maml.py --seed 2 --test --test_way old_user &
#CUDA_VISIBLE_DEVICES=1 python maml.py --seed 2 --test --test_way new_user &

CUDA_VISIBLE_DEVICES=1 python maml.py --seed 3 --test --test_way old_user &
#CUDA_VISIBLE_DEVICES=1 python maml.py --seed 3 --test --test_way new_user &

CUDA_VISIBLE_DEVICES=1 python maml.py --seed 4 --test --test_way old_user &
#CUDA_VISIBLE_DEVICES=1 python maml.py --seed 4 --test --test_way new_user &

CUDA_VISIBLE_DEVICES=1 python maml.py --seed 5 --test --test_way old_user &
#CUDA_VISIBLE_DEVICES=1 python maml.py --seed 5 --test --test_way new_user &

'''
CUDA_VISIBLE_DEVICES=0 python maml.py --seed 0 &
CUDA_VISIBLE_DEVICES=0 python maml.py --seed 1 &
CUDA_VISIBLE_DEVICES=1 python maml.py --seed 2 &
CUDA_VISIBLE_DEVICES=0 python maml.py --seed 3 &
CUDA_VISIBLE_DEVICES=1 python maml.py --seed 4 &
CUDA_VISIBLE_DEVICES=1 python maml.py --seed 5 &
'''
