import argparse

import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--out", type=str, default="factor.pt")
    parser.add_argument("ckpt", type=str)

    args = parser.parse_args()

    ckpt = torch.load(args.ckpt)
    modulate = {
        k: v
        for k, v in ckpt["g_ema"].items()
        if "modulation" in k and "to_rgbs" not in k and "weight" in k
    }

    weight_mat = []
    for k, v in modulate.items():
        weight_mat.append(v)

    W = torch.cat(weight_mat, 0)
    eigvec = torch.svd(W).V.to("cpu")

    torch.save({"ckpt": args.ckpt, "eigvec": eigvec}, args.out)

