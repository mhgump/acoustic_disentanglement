""" Raw audio feature extractors

* MFCC and Melspec based on https://github.com/wnhsu/ScalableFHVAE/
"""
import librosa
import numpy as np
import scipy
import crepe
import pysptk

from speech_representations.features.feature_registry import FeatureSpec, transformer, register_feature
from speech_representations.features.raw_audio import AudioArguments

def _melspec_audio(melspec, sr, num_fft, win_length, hop_length, stft_window):
    return librosa.feature.inverse.mel_to_audio(melspec.transpose(), 
        sr=sr, n_fft=num_fft, hop_length=hop_length, win_length=win_length, window=stft_window)

melspec_audio = FeatureSpec('melspec_audio', _melspec_audio, inputs=['melspec'], tags=['reconstructed_audio'])


def _mfcc_audio(mfcc,  sr, num_fft, num_mels, win_length, hop_length, stft_window):
    return librosa.feature.inverse.mfcc_to_audio(mfcc.transpose(), n_mels=num_mels, sr=sr, n_fft=num_fft, 
        hop_length=hop_length, win_length=win_length, window=stft_window)

mfcc_audio = FeatureSpec('mfcc_audio', _mfcc_audio, inputs=['mfcc'], tags=['reconstructed_audio'])


@transformer(tags=['reconstructed_audio'])
class ReconstructionTransformer:

    arguments_class = AudioArguments

    def __init__(self, feature_name, **kwargs):
        assert feature_name in ['melspec', 'mfcc']
        self.output_feature_name = '{}_audio'.format(feature_name)
        for kwarg_name, kwarg in kwargs.items():
            setattr(self, kwarg_name, kwarg)

    def get_wav(self):
        return self.get(self.output_feature_name)
