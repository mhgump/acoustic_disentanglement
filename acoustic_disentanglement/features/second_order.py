""" Raw audio feature extractors

* MFCC and Melspec based on https://github.com/wnhsu/ScalableFHVAE/
"""
import librosa
import numpy as np
import scipy
import crepe
import pysptk

from speech_representations.features.feature_registry import FeatureSpec


def _mean_formant_difference(formant):
    masks = (formant[:, :-1] > 0) * (formant[:, 1:] > 0)
    total = np.sum(masks * (formant[:, :-1] - formant[:, 1:]))
    N = np.sum(masks)
    return total * np.reciprocal(1. * N)

mean_formant_difference = FeatureSpec('mean_formant_difference', _mean_formant_difference, tags=['second_order'])


def _first_formant_difference(formant):
    mask = (formant[:, 0] > 0) * (formant[:, 1] > 0)
    return (formant[:, 1] - formant[:, 0]) * mask

first_formant_difference = FeatureSpec('first_formant_difference', _first_formant_difference, tags=['second_order'])


# def _pitch(pitch_crepe, pitch_swipe, pitch_rapt):
#     return (pitch_crepe + pitch_swipe + pitch_rapt) * 1. / (pitch_crepe>0 + pitch_swipe>0 + pitch_rapt>0 + 0.0)

# pitch = FeatureSpec('pitch', _pitch, tags=['second_order'])
