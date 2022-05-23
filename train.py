import argparse
from trainer import Trainer


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--logger', type=bool, default=True, help='Enbale wandb logging')
    parser.add_argument('--data_root', type=str, help='Path to dataset')
    parser.add_argument('--frame_stack', type=int, default=5, help='Number of timestamps to process')

    parser.add_argument('--max_iters', type=int, default=100000, help='Maximum training iterations')
    parser.add_argument('--val_rate', type=int, default=10, help='Every k-th step to validate')
    parser.add_argument('--save_rate', type=int, default=100, help='Every k-th step to save models')
    parser.add_argument('--ckpt_root', type=str, help='Path to saving models')
    parser.add_argument('--val_root', type=str, help='Path to saving validation videos')

    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--img_size', type=int, default=128, help='Size of images')
    parser.add_argument('--lr_gen', type=float, default=3e-5, help='Generator LR')
    parser.add_argument('--lr_disc', type=float, default=3e-5, help='Discriminator LR')
    parser.add_argument('--wd', type=float, default=1e-7, help='Weight decay')

    parser.add_argument('--flow_ckpt', type=str, help='Path to stored flow checkpoint')

    parser.add_argument('--lambda_anchor', type=float, default=0.5, help='Anchor consistency loss factor')
    parser.add_argument('--exp_name', type=str, help='Name of experiment for logging')
    args = parser.parse_args()
    kwargs = vars(args)
    flowargs = DotDict({'rgb_max': 255.})

    newTrainer = Trainer(flowargs, **kwargs)

    if args.logger:
        cfg = kwargs
        exp_name = args.exp_name
    else:
        cfg = None
        exp_name = 'Default name'
    newTrainer.train(exp_name, cfg)
