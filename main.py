import argparse
from denoiser.train import train
from denoiser.inference import inference

import matplotlib

matplotlib.use("Agg")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument(
        "--model_type", type=str, required=True, help="select from s/b/3"
    )
    parser.add_argument("--train_data", nargs="+")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--test_data", type=str, default="data/Test")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--noise_level", type=int, default=25)
    args = parser.parse_args()

    if args.mode == "train":
        train(vars(args))
    elif args.mode == "test":
        inference(vars(args))


if __name__ == "__main__":
    main()
