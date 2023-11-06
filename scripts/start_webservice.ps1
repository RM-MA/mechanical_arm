python web_service/app_v3_rl.py `
--candidates-size 43 34 24  43 34 18  41 29 19  33 29 20  31 23 16  26 19 13 `
--device 'cuda' `
--preview-capping 8 `
--min-ems-width 0 `
--min-height-diff 20 `
--same-height-threshold 0 `
--real 0 `
--hidden-dim 128 `
--resume-path-capping "log/ppo/tapnet/[100_100]_C_0_20n-10-60-30height-h-diff51-min-ems-10/actor.pth"
# --hidden-dim 256 `
# --resume-path-capping "log/ppo/tapnet/[100_100]_C_0_enc-dec-20n-id-stair/policy.pth"

# --candidates-size 25 36 39  28 36 20  39 53 30  33 43 32  27 35 20  24 34 21  30 33 17  27 32 28  23 63 33  44 46 31  23 28 25  33 47 12  31 38 28  28 40 15  30 32 36 41 77 50 36 65 30 24 32 21 32 47 32 53 76 36 41 45 58 44 79 35 30 39 55 49 78 52 32 79 49 35 43 40 24 34 21 52 54 54 47 50 40 40 44 56 36 47 27 60 73 33 60 61 30 24 25 31 45 63 28 `
# --candidates-size 25 36 39  28 36 20  39 53 30  33 43 32  27 35 20  24 34 21  30 33 17  27 32 28  23 63 33  44 46 31  23 28 25  33 47 12  31 38 28  28 40 15  30 32 36 41 77 50 36 65 30 24 32 21 32 47 32 53 76 36 41 45 58 44 79 35 30 39 55 49 78 52 32 79 49 35 43 40 24 34 21 52 54 54 47 50 40 40 44 56 36 47 27 60 73 33 60 61 30 24 25 31 45 63 28 43 34 24  43 34 18  41 29 19  33 29 20  31 23 16  26 19 13 `
# --resume-path-capping "log/ppo/tapnet/[100_100]_C_0_min_ems_10/policy.pth"
# --resume-path-capping "log/ppo/tapnet/[100_100]_C_0_15n-10-70-30height-h-diff51-min-ems-10/checkpoint_3.pth"
# --resume-path-capping "log/ppo/tapnet/[100_100]_C_0_20n-10-60-height-diff-001-min-ems-10/policy.pth"
# --resume-path-capping "log/ppo/tapnet/[100_100]_C_0_20n-10-60-30height-height-diff-005-min-ems-10/policy.pth" # 不行
# --resume-path-capping "log/ppo/tapnet/[100_100]_C_0_20n-10-60-height-diff-005-min-ems-10-2/policy.pth" # 不行
# --resume-path-capping "log/ppo/tapnet/[100_100]_C_0_20n-10-60-30height-h-diff51-min-ems-10-2/policy.pth" # 还是学到了高度差的
# --resume-path-capping "log/ppo/tapnet/[100_100]_C_0_20n-10-60-height-diff-005-min-ems-10/checkpoint_3.pth"