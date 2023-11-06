python web_service/app_v3_rl.py \
--candidates-size 43 34 24  43 34 18  41 29 19  33 29 20  31 23 16  26 19 13 \
--device 'cuda' \
--preview-capping 8 \
--capping-dilate 3 \
--min-ems-width 15 \
--inflate-size 5 \
--same-height-threshold 1 \
--same-height-threshold-max 4 \
--corner-pos 0 \
--model1 "./cp/ppo_tnpp_none_ems-id-stair_hard_after_pack_rew_H_ctn112_20k_keep/keep.pth" \
--model2 "./cp/ppo_tnpp_none_ems-id-stair_hard_after_pack_rew_H_ctn112_20k_keep/checkpoint_0.pth" \
--model3 "./cp/ppo_tnpp_none_ems-id-stair_hard_after_pack_rew_H_ctn112_20k_keep/checkpoint_1.pth" \
# --model4 "./cp/ppo_tnpp_none_ems-id-stair_hard_after_pack_rew_H_ctn112_20k/policy.pth" \

# --resume-path-capping "log/[100_100]_C_0_enc-dec-20n-id-stair/policy.pth"
# 231009-150353
# 231009-183003