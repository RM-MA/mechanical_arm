
import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--preview-capping", type=int, default=5)
    parser.add_argument("--preview-simu", type=int, default=5)
    parser.add_argument("--simu-len", type=int, default=60)

    parser.add_argument("--capping-method", type=str, default="tap")
    parser.add_argument("--capping-raise-step", type=int, default=30)
    parser.add_argument("--capping-shrink-step", type=int, default=2)

    parser.add_argument("--vision-tolerant", type=int, default=4)
    parser.add_argument("--inflate-size", type=float, default=0)
    parser.add_argument("--capping-dilate", type=float, default=2.0)
    parser.add_argument("--corner-pos", type=int, default=0)
    parser.add_argument("--min-ems-width", type=int, default=15)
    parser.add_argument("--min-height-diff", type=int, default=2)
    parser.add_argument("--same-height-threshold", type=float, default=1.0)
    parser.add_argument("--same-height-threshold-max", type=float, default=2.0)
    parser.add_argument("--gripper-size", type=int, nargs='*', default=None)
    
    parser.add_argument("--real", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=128)

    parser.add_argument("--container-size", type=int, nargs="*", default=[120, 100, 120])
    parser.add_argument("--candidates-size", type=int, nargs="*", default=None)
    parser.add_argument("--fixed-sequence", type=int, nargs="*", default=None)

    parser.add_argument("--test-json", type=str, nargs="*", default=None)

    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--test-num", type=int, default=10)
    
    parser.add_argument( "--device", type=str, default="cuda" )
    parser.add_argument("--model1", type=str, default="./cp/ppo_tnpp_none_ems-id-stair_hard_after_pack_rew_H_ctn112_20k_keep/keep.pth")
    parser.add_argument("--model2", type=str, default=None)
    parser.add_argument("--model3", type=str, default=None)
    parser.add_argument("--model4", type=str, default=None)

    args = parser.parse_args()
    if args.candidates_size != None:
        args.item_size_set = []
        for i in range(0, len(args.candidates_size), 3):
            args.item_size_set.append([args.candidates_size[i], args.candidates_size[i+1], args.candidates_size[i+2]])

        print(args.item_size_set)

        if args.fixed_sequence != None:
            assert len(args.fixed_sequence) == len(args.item_size_set)

    else:
        args.item_size_set = None

    return args

# _vision_tolerant = 4
# _capping_dilate = 2.0
# _inflate_size = 3
# _corner_pos = 0