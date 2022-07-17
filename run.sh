'''
#CUDA_VISIBLE_DEVICES=0 python maml.py --model_name ColReg --seed 0 --lambd 1
CUDA_VISIBLE_DEVICES=0 python maml.py --model_name ColReg --seed 0 --lambd 2 &
CUDA_VISIBLE_DEVICES=0 python maml.py --model_name ColReg --seed 0 --lambd 3 &
CUDA_VISIBLE_DEVICES=1 python maml.py --model_name ColReg --seed 0 --lambd 4 &
CUDA_VISIBLE_DEVICES=1 python maml.py --model_name ColReg --seed 0 --lambd 5 &
CUDA_VISIBLE_DEVICES=1 python maml.py --model_name ColReg --seed 0 --lambd 6 &
wait
CUDA_VISIBLE_DEVICES=0 python maml.py --model_name ColReg --seed 0 --lambd 7 &
CUDA_VISIBLE_DEVICES=0 python maml.py --model_name ColReg --seed 0 --lambd 8 &
CUDA_VISIBLE_DEVICES=1 python maml.py --model_name ColReg --seed 0 --lambd 9 &
CUDA_VISIBLE_DEVICES=1 python maml.py --model_name ColReg --seed 0 --lambd 10 &
wait

CUDA_VISIBLE_DEVICES=0 python maml.py --seed 0 --model_name ColReg --lambd 1 --test --test_way  old_user &
CUDA_VISIBLE_DEVICES=0 python maml.py --seed 0 --model_name ColReg --lambd 1 --test --test_way  new_user &
CUDA_VISIBLE_DEVICES=1 python maml.py --seed 0 --model_name ColReg --lambd 2 --test --test_way  old_user &
CUDA_VISIBLE_DEVICES=1 python maml.py --seed 0 --model_name ColReg --lambd 2 --test --test_way  new_user &
wait

CUDA_VISIBLE_DEVICES=0 python maml.py --seed 0 --model_name ColReg --lambd 3 --test --test_way  old_user &
CUDA_VISIBLE_DEVICES=0 python maml.py --seed 0 --model_name ColReg --lambd 3 --test --test_way  new_user &
CUDA_VISIBLE_DEVICES=1 python maml.py --seed 0 --model_name ColReg --lambd 4 --test --test_way  old_user &
CUDA_VISIBLE_DEVICES=1 python maml.py --seed 0 --model_name ColReg --lambd 4 --test --test_way  new_user &
wait
CUDA_VISIBLE_DEVICES=0 python maml.py --seed 0 --model_name ColReg --lambd 5 --test --test_way  old_user &
CUDA_VISIBLE_DEVICES=0 python maml.py --seed 0 --model_name ColReg --lambd 5 --test --test_way  new_user &
CUDA_VISIBLE_DEVICES=1 python maml.py --seed 0 --model_name ColReg --lambd 6 --test --test_way  old_user &
CUDA_VISIBLE_DEVICES=1 python maml.py --seed 0 --model_name ColReg --lambd 6 --test --test_way  new_user &
wait

CUDA_VISIBLE_DEVICES=0 python maml.py --seed 0 --model_name ColReg --lambd 7 --test --test_way  old_user &
CUDA_VISIBLE_DEVICES=0 python maml.py --seed 0 --model_name ColReg --lambd 7 --test --test_way  new_user &
CUDA_VISIBLE_DEVICES=1 python maml.py --seed 0 --model_name ColReg --lambd 8 --test --test_way  old_user &
CUDA_VISIBLE_DEVICES=1 python maml.py --seed 0 --model_name ColReg --lambd 8 --test --test_way  new_user &
wait

CUDA_VISIBLE_DEVICES=0 python maml.py --seed 0 --model_name ColReg --lambd 9 --test --test_way  old_user &
CUDA_VISIBLE_DEVICES=0 python maml.py --seed 0 --model_name ColReg --lambd 9 --test --test_way  new_user &
CUDA_VISIBLE_DEVICES=1 python maml.py --seed 0 --model_name ColReg --lambd 10 --test --test_way  old_user &
CUDA_VISIBLE_DEVICES=1 python maml.py --seed 0 --model_name ColReg --lambd 10 --test --test_way  new_user &
wait





CUDA_VISIBLE_DEVICES=1 python maml.py --model_name ColReg --seed 1 --rerun &
CUDA_VISIBLE_DEVICES=1 python maml.py --model_name ColReg --seed 2 --rerun &
CUDA_VISIBLE_DEVICES=0 python maml.py --model_name ColReg --seed 3 --rerun &
CUDA_VISIBLE_DEVICES=0 python maml.py --model_name ColReg --seed 4 --rerun &
CUDA_VISIBLE_DEVICES=1 python maml.py --model_name ColReg --seed 5 --rerun &

CUDA_VISIBLE_DEVICES=1 python maml.py --seed 0 --model_name ColReg --test --test_way  old_user &
CUDA_VISIBLE_DEVICES=1 python maml.py --seed 0 --model_name ColReg --test --test_way  new_user &

CUDA_VISIBLE_DEVICES=0 python maml.py --seed 1 --model_name ColReg --test --test_way  old_user &
CUDA_VISIBLE_DEVICES=0 python maml.py --seed 1 --model_name ColReg --test --test_way  new_user &
wait
CUDA_VISIBLE_DEVICES=1 python maml.py --seed 2 --model_name ColReg --test --test_way  old_user &
CUDA_VISIBLE_DEVICES=1 python maml.py --seed 2 --model_name ColReg --test --test_way  new_user &

CUDA_VISIBLE_DEVICES=0 python maml.py --seed 3 --model_name ColReg --test --test_way  old_user &
CUDA_VISIBLE_DEVICES=0 python maml.py --seed 3 --model_name ColReg --test --test_way  new_user &
wait

CUDA_VISIBLE_DEVICES=1 python maml.py --seed 4 --model_name ColReg --test --test_way  old_user &
CUDA_VISIBLE_DEVICES=1 python maml.py --seed 4 --model_name ColReg --test --test_way  new_user &

CUDA_VISIBLE_DEVICES=0 python maml.py --seed 5 --model_name ColReg --test --test_way  old_user &
CUDA_VISIBLE_DEVICES=0 python maml.py --seed 5 --model_name ColReg --test --test_way  new_user &

wait
CUDA_VISIBLE_DEVICES=0 python maml.py  --seed 0 &
CUDA_VISIBLE_DEVICES=1 python maml.py  --seed 1 &
CUDA_VISIBLE_DEVICES=1 python maml.py  --seed 2 &
CUDA_VISIBLE_DEVICES=0 python maml.py  --seed 3 &
CUDA_VISIBLE_DEVICES=0 python maml.py  --seed 4 &
CUDA_VISIBLE_DEVICES=1 python maml.py  --seed 5 &

CUDA_VISIBLE_DEVICES=0 python maml.py --model_name PD --seed 0 &
CUDA_VISIBLE_DEVICES=1 python maml.py --model_name PD --seed 1 &
CUDA_VISIBLE_DEVICES=1 python maml.py --model_name PD --seed 2 &
CUDA_VISIBLE_DEVICES=0 python maml.py --model_name PD --seed 3 &
CUDA_VISIBLE_DEVICES=0 python maml.py --model_name PD --seed 4 &
CUDA_VISIBLE_DEVICES=1 python maml.py --model_name PD --seed 5 &


CUDA_VISIBLE_DEVICES=1 python maml.py --seed 1 --test --model_name PD --test_way  new_user &
CUDA_VISIBLE_DEVICES=0 python maml.py --seed 2 --test --model_name PD --test_way  new_user &

CUDA_VISIBLE_DEVICES=1 python maml.py --seed 1 --test --model_name PD --test_way  old_user &
CUDA_VISIBLE_DEVICES=0 python maml.py --seed 2 --test --model_name PD --test_way  old_user &

wait

CUDA_VISIBLE_DEVICES=1 python maml.py --seed 1 --test  --test_way  new_user &
CUDA_VISIBLE_DEVICES=0 python maml.py --seed 2 --test  --test_way  new_user &

CUDA_VISIBLE_DEVICES=1 python maml.py --seed 1 --test  --test_way  old_user &
CUDA_VISIBLE_DEVICES=0 python maml.py --seed 2 --test  --test_way  old_user &

CUDA_VISIBLE_DEVICES=0 python maml.py --seed 0 &
CUDA_VISIBLE_DEVICES=0 python maml.py --seed 1 &
CUDA_VISIBLE_DEVICES=1 python maml.py --seed 2 &
CUDA_VISIBLE_DEVICES=0 python maml.py --seed 3 &
CUDA_VISIBLE_DEVICES=1 python maml.py --seed 4 &
CUDA_VISIBLE_DEVICES=1 python maml.py --seed 5 &
'''


'''
CUDA_VISIBLE_DEVICES=0 python maml.py --model_name ColMU --seed 0 --meta_hid 32 --rerun &
CUDA_VISIBLE_DEVICES=1 python maml.py --model_name ColMSU --seed 0 --meta_hid 32 --rerun &

wait
CUDA_VISIBLE_DEVICES=1 python maml.py --seed 0 --model_name ColMU --meta_hid 32 --test --test_way  old_user &
CUDA_VISIBLE_DEVICES=1 python maml.py --seed 0 --model_name ColMU --meta_hid 32 --test --test_way  new_user &

CUDA_VISIBLE_DEVICES=1 python maml.py --seed 0 --model_name ColMSU --meta_hid 32 --test --test_way  old_user &
CUDA_VISIBLE_DEVICES=1 python maml.py --seed 0 --model_name ColMSU --meta_hid 32 --test --test_way  new_user &
'''


CUDA_VISIBLE_DEVICES=0 python maml.py --seed 0 --model_name ColMS --meta_hid 32 --test --test_way  new_user
CUDA_VISIBLE_DEVICES=0 python maml.py --seed 0 --model_name ColMS --meta_hid 32 --test --test_way  old_user 
