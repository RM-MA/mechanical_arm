
--box-num 20 --box-range 10 60 --container-size 100 100 100 `

--resume-path ".\log_stable\tap_fake-rand\ppo\tnpp_attn_xyz_100_100_10-60_20_ems-id-stair_space_limit_train-4corner\checkpoint_0.pth" `
--note 'test' `

# last
# normal C
# 0.79, 3.43 greedy
# 0.804, 3.41, NA

# ctnH C
# 0.705, 3.58 greedy
# 0.732, 3.44 C
# 0.732, 3.44 N

# 0.726, 3.47 NA-ctnH

# all
# 0.713, 3.54 greedy
# 0.738, 3.41 C
# 0.738, 3.40 N

# 0.743, 3.39 NA


# old C
# last
# 0.775  1.84 greedy
# 0.676  1.52  # 1~6

# 0.717  2.07  # 2~6


--box-num 10 --box-range 1 8 --container-size 10 10 10 `
--box-num 10 --box-range 2 16 --container-size 20 20 20 `

python tap_train.py `
--box-num 10 --box-range 10 80 --container-size 100 100 100 `
--train 1 `
--test-num 1 `
--model 'tnpp' `
--prec-type 'attn' `
--fact-type 'tap_fake' `
--data-type 'ppsg' `
--ems-type 'ems-id-stair' `
--stable-rule 'hard_after_pack' `
--rotate-axes 'x' 'y' 'z' `
--hidden-dim 128 `
--world-type 'real' `
--container-type 'single' `
--pack-type 'last' `
--stable-predict 1 `
--note 'train2' `
--reward-type 'C' `

--max-epoch 70 `
--resume-path "log\result\z_20_20_2-16_10_tap_fake_ppsg\real_multi_last\ppo_tnpp_attn_ems-id-stair_hard_after_pack_train_pred\policy.pth" `

--resume-path "log\result\xyz_100_100_10-80_20_tap_fake_rand\ideal_single_last\ppo_tnpp_attn_ems-id-stair_none_train\policy.pth" `
--max-epoch 50 `


--seed 777 `



python tap_train.py `
--box-num 20 --box-range 1 8 --container-size 10 10 10 `
--train 1 `
--test-num 1 `
--model 'tnpp' `
--prec-type 'attn' `
--fact-type 'tap_fake' `
--data-type 'rand' `
--ems-type 'ems-id-stair' `
--stable-rule 'none' `
--rotate-axes 'x' 'y' 'z' `
--hidden-dim 128 `
--world-type 'ideal' `
--container-type 'multi' `
--pack-type 'all' `
--stable-predict 0 `
--note 'train' `



# ===================================================


python tap_train.py `
--box-num 10 --box-range 10 80 --container-size 100 100 100 `
--train 0 `
--test-num 200 `
--model 'greedy' `
--prec-type 'attn' `
--fact-type 'tap_fake' `
--data-type 'ppsg' `
--ems-type 'ems-id-stair' `
--stable-rule 'hard_after_pack' `
--rotate-axes 'z' `
--hidden-dim 128 `
--world-type 'real' `
--container-type 'multi' `
--pack-type 'last' `
--stable-predict 0 `
--note 'train' `
--reward-type 'C' `

# 


python tap_train.py `
--box-num 20 --box-range 10 80 --container-size 100 100 100 `
--train 0 `
--test-num 200 `
--model 'tnpp' `
--prec-type 'attn' `
--fact-type 'tap_fake' `
--data-type 'fix' `
--ems-type 'ems-id-stair' `
--stable-rule 'hard_after_pack' `
--rotate-axes 'x' 'y' 'z' `
--hidden-dim 128 `
--world-type 'ideal' `
--container-type 'multi' `
--pack-type 'all' `
--stable-predict 0 `
--note 'train' `
--resume-path "./log/result/xyz_100_100_10-80_20_tap_fake_fix/ideal_multi_all/ppo_tnpp_attn_ems-id-stair_none_train/policy.pth"

--resume-path "./log/result/xyz_100_100_10-80_20_tap_fake_rand/real_single_last/ppo_tnpp_attn_ems-id-stair_hard_after_pack_train-keep_pred/policy.pth"

--max-epoch 50 `
--reward-type 'C' `



# 

python tap_train.py `
--box-num 20 --box-range 10 80 --container-size 100 100 100 `
--train 1 `
--test-num 1 `
--model 'tnpp' `
--prec-type 'attn' `
--fact-type 'tap_fake' `
--data-type 'fix' `
--ems-type 'ems-id-stair' `
--stable-rule 'hard_after_pack' `
--rotate-axes 'x' 'y' 'z' `
--hidden-dim 128 `
--world-type 'real' `
--container-type 'single' `
--pack-type 'all' `
--stable-predict 1 `
--note 'train' `

--resume-path "./log/result/xyz_100_100_10-80_20_tap_fake_fix/real_single_last/ppo_tn_cnn_ems-id-stair_hard_after_pack_train/policy.pth"


new todo


1. box num

30
    single **** no
    multi-all
    multi-last  
40
    single **** no
    multi-all
    multi-last


2. feasibile mask
no mask ... *
soft CS ... *

3. corner num
all corner ... < 

4. encoder
cnn ... *
rnn ... < *

5. location
ems ... *


python tap_train.py `
--box-num 40 --box-range 10 80 --container-size 100 100 100 `
--train 1 `
--test-num 1 `
--model 'tnpp' `
--prec-type 'attn' `
--fact-type 'tap_fake' `
--data-type 'rand' `
--ems-type 'ems-id-stair' `
--stable-rule 'hard_after_pack' `
--rotate-axes 'x' 'y' 'z' `
--hidden-dim 128 `
--world-type 'real' `
--container-type 'multi' `
--pack-type 'last' `
--stable-predict 1 `
--reward-type 'C' `
--note 'train' `
--require-box-num 20 `

