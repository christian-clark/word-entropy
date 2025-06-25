import argparse
import numpy as np
import pandas as pd


def main(args):
    resamples = args.resamples
    data = pd.read_csv(args.data, header=None).values
    start = args.startsample
    if start is None:
        start = 0
    end = args.endsample
    if end is None:
        end = data.shape[1]
    data = data[:, start:end]
    words, samples = data.shape
    # entropies across all words and resamples
    all_resampled_entropies = np.empty(shape=(words, 0))
    for i in range(resamples):
        ixs = np.random.choice(samples, size=(words, samples), replace=True)
        resampled_data = np.take_along_axis(data, indices=ixs, axis=1)
        # entropies across all words in one resample
        resampled_entropies = np.mean(resampled_data, axis=1)
        resampled_entropies = np.expand_dims(resampled_entropies, axis=1)
        all_resampled_entropies = np.concat(
            [all_resampled_entropies, resampled_entropies],
            axis=1
        )
    means = np.mean(all_resampled_entropies, axis=1)
    sds = np.std(all_resampled_entropies, axis=1)
    cvs = sds / means
    avg_cv = np.mean(cvs)
    print("------------------------")
    print("number of words:", words)
    print("number of original samples:", samples)
    print("number of resamples:", resamples)
    print("average coefficient of variation:", avg_cv)
    print("------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("--resamples", "-r", type=int, default=1000)
    parser.add_argument("--startsample", "-s", type=int, default=None)
    parser.add_argument("--endsample", "-e", type=int, default=None)
    args = parser.parse_args()
    main(args)