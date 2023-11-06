echo $0
cat $0

python web_service/app_v3_rl.py \
--candidates-size 53 44 21   55 32 37   55 32 32  55 32 24  55 32 20  43 34 24   43 34 18   41 29 19   31 23 16  \
--device 'cuda' \
--capping-dilate 2.0 \
--preview-capping 8 \
--min-ems-width 20 \
--min-height-diff 0 \
--same-height-threshold 0.0 \
--hidden-dim 256 \
--resume-path-capping "log/ppo/tapnet/[100_100]_C_0_enc-dec-20n-id-stair/policy.pth"
# --hidden-dim 128 \
# --resume-path-capping "log/[100_100]_C_0_20n-10-60-30height-h-diff51-min-ems-10/actor.pth"