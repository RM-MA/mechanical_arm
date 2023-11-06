python web_service/app_v3_rl.py `
--candidates-size 43 34 24  43 34 18  41 29 19  33 29 20  31 23 16  26 19 13 `
--device 'cuda' `
--preview-capping 8 `
--min-ems-width 15 `
--min-height-diff 0 `
--inflate-size 0 `
--corner-pos 0 `
--same-height-threshold 0 `
--model1 "./cp/ppo_tnpp_none_ems-id-stair_hard_after_pack_rew_H_ctn112_20k_keep/keep.pth" `
--model2 "./cp/ppo_tnpp_none_ems-id-stair_hard_after_pack_rew_H_ctn112_20k_keep/checkpoint_2.pth" `
--model3 "./cp/ppo_tnpp_none_ems-id-stair_hard_after_pack_rew_H_ctn112_20k/policy.pth" `
# --model1 "./cp/ppo_tnpp_none_ems-id-stair_hard_after_pack_rew_H_ctn112_20k_keep/keep.pth" `
# --model1 "./cp/ppo_tnpp_none_ems-id-stair_hard_after_pack_rew_H_ctn112_20k_keep/policy.pth" `
# --model2 "./cp/ppo_tnpp_none_ems-id-stair_hard_after_pack_rew_H_ctn112_20k_keep/checkpoint_0.pth" `


# --resume-path-capping "./cp/ppo_tnpp_none_ems-id-stair_hard_after_pack_rew_HD_60d_pred/policy.pth" # 挺好
# --resume-path-capping "./log/result/z_100_100_10-70_20_box_rand/real_multi_all/ppo_tnpp_none_ems-id-stair_hard_after_pack_rew_HD_60d_1_pred/policy.pth" # 不行
# --resume-path-capping "./log/result/z_100_100_10-70_20_box_rand/real_multi_all/ppo_tnpp_none_ems-id-stair_hard_after_pack_rew_H_2init/policy.pth" # 不行
# --resume-path-capping "./log/result/z_100_100_10-70_20_box_rand/real_multi_all/ppo_tnpp_none_ems-id-stair_hard_after_pack_rew_HD_60d_1/policy.pth" # 不行
# --resume-path-capping "./log/result/z_100_100_10-70_20_box_rand/real_multi_all/ppo_tnpp_none_ems-id-stair_hard_after_pack_rew_HD_60d_1_2init/policy.pth" # 不行
# --resume-path-capping "./log/result/z_100_100_10-70_20_box_rand/real_multi_all/ppo_tnpp_none_ems-id-stair_hard_after_pack_rew_HD_60d_2_2init/policy.pth" # 挺好吗？
# --resume-path-capping "./log/result/z_100_100_10-70_20_box_rand/real_multi_all/ppo_tnpp_none_ems-id-stair_hard_after_pack_rew_HD_60d_2/policy.pth" # 不行
# --resume-path-capping "./log/result/z_100_100_10-70_20_box_rand/real_multi_all/ppo_tnpp_none_ems-id-stair_hard_after_pack_rew_H/policy.pth" # 不行
# --resume-path-capping "./log/result/z_100_100_10-60_20_box_rand/real_single_last/ppo_tnpp_none_ems-id-stair_hard_after_pack_rew_H/policy.pth"
# --resume-path-capping "./log/result/z_100_100_10-60_20_box_rand/real_multi_all/ppo_tnpp_none_ems-id-stair_hard_before_pack_rew_H_2init/policy.pth" # 不咋地
# --resume-path-capping "./log/result/z_100_100_10-60_20_box_rand/real_multi_all/ppo_tnpp_none_ems-id-stair_hard_after_pack_rew_H_2init/policy.pth"
# --resume-path-capping "./log/result/z_100_100_10-60_20_box_rand/real_single_all/ppo_tnpp_none_ems-id-stair_hard_after_pack_rew_H_1init/policy.pth"
# --resume-path-capping "./log/result/z_100_100_10-60_20_box_rand/real_single_last/ppo_tnpp_none_ems-id-stair_hard_after_pack_rew_C_3/policy.pth"

# python tap_train.py --reward H --note debug --train 0 --test 20 `
# --box-range 10 60 --box-num 70 --stable-rule hard_before_pack `
# --save 1 --stable-predict 0 --fact-type box --world-type real --container-type multi --pack-type all --init-ctn-num 1 `
# --resume-path ".\log\result\z_100_100_10-60_50_box_rand\real_multi_all\ppo_tnpp_none_ems-id-stair_hard_after_pack_rew_H_ctn112_20k\policy.pth" --gripper-size 40 40 40 `   
# --container-size 100 100 150