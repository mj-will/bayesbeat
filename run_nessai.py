#!/usr/bin/env python
import argparse
import os
import shutil
import logging

import matplotlib.pyplot as plt
import seaborn as sns
from nessai import config
from nessai.model import Model
from nessai.flowsampler import FlowSampler
from nessai.utils import setup_logger
from nessai.plot import corner_plot
from nessai.livepoint import live_points_to_dict
import numpy as np
import hdf5storage

sns.set_palette("colorblind")
sns.set_style("ticks")

logger = logging.getLogger("nessai")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing results directory.")
    parser.add_argument("--seed", default=1234, type=int, help="Random seed.")
    parser.add_argument("--datafile", default=None, type=str, help="Data file to load.")
    parser.add_argument("--index", default=None, type=int, help="Index of the data in the data file.")
    parser.add_argument("--label", default=None, type=str, help="Label added to the end of output directory.")
    parser.add_argument("--n-pool", default=None, type=int, help="Number of cores to use.")
    parser.add_argument("--log-level", default=None, type=str, help="Logging level")
    return parser.parse_args()


def sigmodel(A1, A2, t1, t2, p1, dp, dw, x):
    """Signal model"""
    Ap = A1 * np.exp(-x / t1)
    Bp = A2 * np.exp(-x / t2)
    p2 = p1 - dp
    App = -Ap * np.sin(p1) - Bp * np.sin(dw * x + p2)
    Bpp = Ap * np.cos(p1) + Bp * np.cos(dw * x + p2)

    return np.sqrt(App**2 + Bpp**2)


def signal_from_dict(d, x_data):
    return sigmodel(
        d["A1"], d["A2"], d["t1"], d["t2"], d["p1"], d["dp"], d["dw"], x_data
    )


class DoubleDecayingModel(Model):
    def __init__(self, x, y):
        # define param names as list
        self.names = ["A1", "t1", "t2", "p1", "dp", "dw", "sigma"]
        self.bounds = {
            "A1": (0.5, 1),
            "t1": (10, 10000),
            "t2": (10, 10000),
            "p1": (0, 2 * np.pi),
            "dp": (0, 2 * np.pi),
            "dw": (0, 1),
            "sigma": (0, 1),
        }
        self.x_data = x
        self.y_data = y
        self.n_samples = len(x)

    def log_prior(self, x):
        """
        Returns log of prior given a live point assuming uniform
        priors on each parameter.
        """
        log_p = np.log(self.in_bounds(x), dtype="float")
        for n in self.names:
            log_p -= np.log(self.bounds[n][1] - self.bounds[n][0])
        return log_p

    def log_likelihood(self, x):

        sig = x["sigma"]
        norm_const = -0.5 * self.n_samples * np.log(2 * np.pi * sig**2)

        y_signal = sigmodel(
            x["A1"],
            1 - x["A1"],
            x["t1"],
            x["t2"],
            x["p1"],
            x["dp"],
            x["dw"],
            self.x_data,
        )

        lik_func = norm_const + np.sum(
            -0.5 * ((self.y_data - y_signal) ** 2 / (sig**2))
        )
        return lik_func


def get_simulated_data(rng):
    fs = 2.0
    max_t_obs = 3600.0

    t_obs = rng.uniform(60, max_t_obs)
    n_samples = int(fs * t_obs)

    # Scale of the noise
    noise_scale = rng.uniform(0, 0.1)
    # Amplitudes
    A1 = rng.uniform(0.5, 1.0)
    A2 = 1 - A1

    truth = {
        "A1": A1,
        "A2": A2,
        "t1": rng.uniform(10, 5000),
        "t2": rng.uniform(10, 5000),
        "p1": rng.uniform(-np.pi / 2, np.pi / 2),
        "dp": rng.uniform(-np.pi / 2, np.pi / 2),
        "dw": rng.uniform(0, 1.0),
        "sigma": noise_scale,
    }

    x_data = np.linspace(0, t_obs, n_samples)
    true_signal = signal_from_dict(truth, x_data)

    noise = rng.normal(loc=0.0, scale=noise_scale, size=n_samples)
    logger.info("Standard deviation of noise:")
    logger.info(f"True value: {noise_scale}")
    logger.info(f"Estimated: {np.std(noise)}")

    y_data = true_signal + noise
    return x_data, y_data, truth


def get_data(path: str, index: int):

    if index is None:
        raise ValueError("Must specify index")

    matdata = hdf5storage.loadmat(path)
    times = matdata["ring_times"].T
    amplitudes = matdata["ring_amps"].T
    freqs = matdata["freq"]

    times = times[index]
    amplitudes = amplitudes[index]

    keep = ~np.isnan(times)
    times = times[keep]
    amplitudes = amplitudes[keep]

    x, y = times, amplitudes / amplitudes.max()
    return x, y, freqs[index]


def main():
    """Run nessai"""

    args = parse_args()
    rng = np.random.default_rng(seed=args.seed)

    if args.datafile is None:
        name = f"injection_seed{args.seed}"
    else:
        if args.index is None:
            raise ValueError("Missing index")
        name = f"data_index_{args.index}"
    if args.label is not None:
        name += f"_{args.label}"
    output = os.path.join("outdir", name)

    if args.overwrite and os.path.exists(output):
        shutil.rmtree(output)


    logging_kwargs = {}
    if args.log_level:
        logging_kwargs["log_level"] = args.log_level.upper()

    logger = setup_logger(output=output, **logging_kwargs)

    if args.datafile is None:
        x_data, y_data, truth = get_simulated_data(rng)
        frequency = 2800
    else:
        x_data, y_data, frequency = get_data(args.datafile, args.index)
        logger.info(f"Real data frequency: {frequency}")
        truth = None

    fig = plt.figure(dpi=200)
    plt.scatter(x_data, y_data, label="Data", s=2.0, color="grey")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    fig.savefig(os.path.join(output, "data.png"))

    model = DoubleDecayingModel(x_data, y_data)
    model.truth = truth

    sampler = FlowSampler(
        model,
        output=output,
        nlive=1000,
        n_pool=args.n_pool,
        resume=not args.overwrite,
        seed=args.seed,
        reset_flow=16,
    )
    sampler.run(plot_posterior=False, plot_logXlogL=False)

    if truth:
        truths_corner = [v for k, v in truth.items() if k in model.names]
    else:
        truths_corner = None

    logger.info(f"Parameters of the injected signal: {truth}")
    corner_plot(
        sampler.posterior_samples,
        include=model.names,
        truth=truths_corner,
        filename=os.path.join(output, "corner.png"),
    )

    fit_params = {n: np.median(sampler.posterior_samples[n]) for n in model.names}
    logger.info(f"Recovered values: {fit_params}")
    logger.info(f"phi_1: {1 / (fit_params['t1'] * np.pi * frequency)}")
    logger.info(f"phi_2: {1 / (fit_params['t2'] * np.pi * frequency)}")
    fit_params["A2"] = 1 - fit_params["A1"]

    fit = signal_from_dict(fit_params, x_data)

    fig, axs = plt.subplots(2, 1, dpi=200, sharex=True, figsize=(8, 8))
    axs[0].scatter(x_data, y_data, label="Data", s=2.0, color="grey")
    axs[0].plot(x_data, fit, label="Fit", color="C0", ls="-")
    axs[0].set_ylabel("Amplitude")
    axs[0].legend()

    res = y_data - fit
    logger.info(f"Standard deviation of the residuals: {np.std(res)}")

    axs[1].scatter(x_data, res, label="Data - fit", color="C0", s=2.0)
    axs[1].legend()
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Residuals")

    plt.tight_layout()

    fig.savefig(os.path.join(output, "fit.png"))

    all_params = live_points_to_dict(sampler.posterior_samples)
    all_params["A2"] = 1 - all_params["A1"]
    all_params["phi1"] = 1 / (all_params["t1"] * np.pi * frequency)
    all_params["phi2"] = 1 / (all_params["t2"] * np.pi * frequency)

    header = "\t".join(["parameter", "median", "16", "84", "minus", "plus"])
    values = []
    exclude = config.NON_SAMPLING_PARAMETERS
    for name, post in all_params.items():
        if name in exclude:
            continue
        q50, q16, q84 = \
            np.quantile(post, q=[0.5, 0.16, 0.84])
        plus = q84 - q50
        minus = q50 - q16
        values.append([name, str(q50), str(q16), str(q84), str(minus), str(plus)])
    
    with open(os.path.join(output, "result.txt"), "w") as fp:
        fp.write(header + "\n")
        for v in values:
            fp.write("\t".join(v) + "\n")
    

if __name__ == "__main__":
    main()
