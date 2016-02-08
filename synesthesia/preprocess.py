""" Contains functions for preprocessing audio signals. """

import numpy as np
import pandas as pd


def down_mix(x):
    """ Performs down mixing on the audio signal. Reduces
        multi-channel audio signals into one channel. It
        reduces this by taking the mean across all channels
        into one.

        :param x: the audio signal of shape N x C, where N
            is the number of samples, and C is the number of
            channels

        :return: an audio signal of shape N x 1, where N
            is the number of samples. """
    return np.mean(x, axis=1)


def down_sample(x, sample_rate, k=2):
    """ Performs down sampling on the audio signal. It takes
        ever kth sample of the signal and returns the resulting
        audio signal and the resulting sample rate.

        :param x: the audio signal of shape N x C, where N
            is the number of samples, and C is the number of
            channels

        :param k: the number of every k samples to return

        :return: a tuple of sample rate and the audio signal
            down-sampled to include ever kth sample. """
    if len(x.shape[0]) < 2:
        return sample_rate / k, x[::k]
    return sample_rate / k, x[:, ::k]


def normalize(x):
    """ Normalizes the amplitude of the audio signal. It
        results in dividing the audio signal by the absolute
        value of the maximum of the audio signal

        :param x: the audio signal of shape N x C, where N
            is the number of samples, and C is the number of
            channels

        :return: a normalized audio signal of shape N x C, where
            N is the number of samples, and C is the number of
            channels """
    if len(x.shape[0]) < 2:
        return x.astype(float) / np.max(np.abs(x))
    return x.astype(float) / np.max(np.abs(x), axis=0)
