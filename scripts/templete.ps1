
# tnpp

python tap_train.py --note 'newc' `
--train 1 `
--box-num 20 --box-range 10 60 --container-size 100 100 100 `
--model 'tnpp' `
--prec-type 'cnn' `
--ems-type 'ems' `
--container-flag 'new' `
--lr 3e-4 `

python tap_train.py --note 'newc' `
--train 1 `
--box-num 20 --box-range 10 60 --container-size 100 100 100 `
--model 'tnpp' `
--prec-type 'attn' `
--ems-type 'ems' `
--container-flag 'new' `
--lr 3e-4 `


python tap_train.py --note 'newc' `
--train 1 `
--box-num 20 --box-range 10 60 --container-size 100 100 100 `
--model 'tnpp' `
--prec-type 'attn' `
--ems-type 'ems-id' `
--container-flag 'new' `
--lr 3e-4 `


python tap_train.py --note 'onec-hdiff' `
--train 1 `
--box-num 20 --box-range 10 60 --container-size 100 100 5000 `
--model 'tnpp' `
--prec-type 'attn' `
--ems-type 'ems-id-stair' `
--container-flag 'stop' `
--lr 3e-4 `


python tap_train.py --note 'newc' `
--train 1 `
--box-num 10 --box-range 1 4 --container-size 5 5 5 `
--model 'tnpp' `
--prec-type 'attn' `
--ems-type 'ems' `
--container-flag 'new' `
--lr 3e-4 `



python tap_train.py --note 'newc' `
--box-num 20 --box-range 10 60 --container-size 100 100 100 `
--train 1 `
--model 'tn' `
--prec-type 'cnn' `
--ems-type 'ems' `
--container-flag 'new' `
--lr 1e-3 `


# test
python tap_train.py `
--train 0 `
--box-num 20 --box-range 10 60 --container-size 100 100 5000 `
--test-num 100 `
--model 'greedy' `
--prec-type 'attn' `
--ems-type 'ems' `
--container-flag 'stop' `
--require-box-num 20 `
--resume-path "./log/tap_fake-rand/ppo/tn_cnn_100_100_10-60_20_ems_train/policy.pth"

--resume-path "./log/tap_fake-rand/ppo/tnpp_attn_100_100_10-60_20_ems-id-stair_train2/policy.pth"



# 
python tap_train.py --note 'onec-hdiff' `
--train 1 `
--box-num 20 --box-range 10 60 --container-size 100 100 5000 `
--model 'tnpp' `
--prec-type 'attn' `
--ems-type 'ems-id-stair' `
--container-flag 'stop' `
--lr 3e-4 `


python tap_train.py --note 'onec' `
--train 1 `
--box-num 20 --box-range 10 60 --container-size 100 100 5000 `
--model 'tn' `
--prec-type 'cnn' `
--ems-type 'ems' `
--container-flag 'stop' `
--lr 3e-4 `


python tap_train.py --note 'onec' `
--train 1 `
--box-num 20 --box-range 10 60 --container-size 100 100 5000 `
--model 'tnpp' `
--prec-type 'attn' `
--ems-type 'ems' `
--container-flag 'stop' `
--lr 3e-4 `

python tap_train.py --note 'train-comp' `
--train 1 `
--box-num 10 --box-range 1 4 --container-size 5 5 50 `
--model 'tnpp' `
--prec-type 'attn' `
--ems-type 'ems' `
--container-flag 'stop' `
--lr 3e-4 `

