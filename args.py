import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch project training')

    parser.add_argument(
        "--multigpu",
        help="use one or more gpu(s)",
        type=int,
        nargs='+',
        default=[0, 1, 2, 3]
    )

    parser.add_argument(
        "--batch_size",
        help="mini-batch size",
        type=int,
        default=256
    )

    parser.add_argument(
        "--epoch",
        help="training epoch",
        type=int,
        default=100
    )

    parser.add_argument(
        "--lr",
        help="learning rate",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--dropout",
        help="dropout rate",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--wd",
        help="weight decay",
        type=float,
        default=0e-2
    )

    parser.add_argument(
        "--maxL2",
        help="max L2 norm allowed (None for not employ)",
        type=float,
        default=None
    )

    parser.add_argument(
        "--HV_dim",
        help="the full dimension of hypervectors",
        type=int,
        default=10000
    )

    parser.add_argument(
        "--dim_rdc",
        help="the reduction rate of HD dimension",
        type=float,
        default=(1 - 1000 / 10000)
    )

    parse = parser.parse_args()

    return parse

args = parse_arguments()
