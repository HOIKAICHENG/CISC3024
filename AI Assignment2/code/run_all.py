"""One-command reproduction of the ABNN CIFAR-10 study."""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)


def smoke_test() -> None:
    from bnl import count_bnl, freeze_except_bnl, replace_bn_with_bnl, set_bnl_stochastic
    from resnet import count_parameters, resnet20

    m = resnet20()
    x = torch.randn(4, 3, 32, 32)
    y = m(x)
    assert y.shape == (4, 10), y.shape
    n_bn = sum(isinstance(mod, torch.nn.BatchNorm2d) for mod in m.modules())
    replace_bn_with_bnl(m)
    assert count_bnl(m) == n_bn and n_bn > 0
    trainable = freeze_except_bnl(m)
    assert all(p.requires_grad for p in trainable)
    assert sum(p.requires_grad for p in m.parameters()) == len(trainable)
    set_bnl_stochastic(m, True)
    a, b = m(x), m(x)
    assert not torch.allclose(a, b), "BNL should be stochastic"
    set_bnl_stochastic(m, False)
    a, b = m(x), m(x)
    assert torch.allclose(a, b)
    loss = torch.nn.functional.cross_entropy(m(x), torch.zeros(4, dtype=torch.long))
    loss.backward()
    print(f"smoke OK  params={count_parameters(resnet20()):,}  BNL={n_bn}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="tiny schedule, smoke-level numbers only")
    p.add_argument("--threads", type=int, default=10)
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    smoke_test()

    from prepare_data import main as prepare
    from train_map import train as train_map
    from train_abnn import train as train_abnn
    from evaluate import evaluate
    from make_figures import main as figures

    print("preparing data ...")
    prepare()

    class NS:
        pass

    map_args = NS()
    map_args.epochs = 2 if args.quick else 20
    map_args.batch_size = 128
    map_args.lr = 0.1
    map_args.wd = 5e-4
    map_args.seed = 0
    map_args.threads = args.threads

    abnn_args = NS()
    abnn_args.ckpt = os.path.join(os.path.dirname(HERE), "results", "map_resnet20.pth")
    abnn_args.members = 2 if args.quick else 3
    abnn_args.epochs = 1 if args.quick else 5
    abnn_args.batch_size = 128
    abnn_args.lr = 1e-3
    abnn_args.wd = 1e-4
    abnn_args.factor = 5.0
    abnn_args.n_prioritized = 4
    abnn_args.noise_std = 0.1
    abnn_args.seed = 0
    abnn_args.threads = args.threads

    ev_args = NS()
    ev_args.map_ckpt = abnn_args.ckpt
    ev_args.members = abnn_args.members
    ev_args.l_samples = 2 if args.quick else 3
    ev_args.noise_std = 0.1
    ev_args.batch_size = 256
    ev_args.threads = args.threads

    t0 = time.time()
    train_map(map_args)
    train_abnn(abnn_args)
    evaluate(ev_args)
    figures()
    print(f"all done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
