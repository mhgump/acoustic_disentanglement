""" Raw audio feature extractors

* MFCC and Melspec based on https://github.com/wnhsu/ScalableFHVAE/
"""
import librosa
import numpy as np
import scipy
import crepe
import pysptk

from speech_representations.utils import Arguments
from speech_representations.features.feature_registry import FeatureSpec
from speech_representations.features.normalization import MaskedMeanVarianceNormalization


def _stft(wav, win_length, hop_length, preemphasis, stft_window, num_fft):
    _wav = wav[:]
    if preemphasis > 1e-12:
        _wav = _wav - preemphasis * np.concatenate([[0], _wav[:-1]], 0)
    stft = librosa.core.stft(_wav, 
        n_fft=num_fft, win_length=win_length, hop_length=hop_length, window=stft_window)
    stft = np.abs(stft)
    return stft

stft = FeatureSpec('stft', _stft, inputs=['wav'])


def _intrmd_melspec(stft, sr, num_fft, hop_length, num_mels, norm_mel, log_mel, log_floor):
    melspec = librosa.feature.melspectrogram(sr=sr, S=stft, n_fft=num_fft, 
            hop_length=hop_length, n_mels=num_mels, norm=norm_mel)
    if log_mel:
        melspec = np.log(melspec)
        melspec[melspec < log_floor] = log_floor
    return melspec

intrmd_melspec = FeatureSpec('intrmd_melspec', _intrmd_melspec, inputs=['stft'])


def _melspec(intrmd_melspec):
    return intrmd_melspec.transpose()

_melspec_length = lambda obj: obj.num_mels
melspec = FeatureSpec('melspec', _melspec, inputs=['intrmd_melspec'], never_store=True, save=True,
    length=_melspec_length, dtype=np.float32, jagged_length=True, tags=['audio_representation', 'raw_audio'])


def _mfcc(intrmd_melspec, sr, num_mfcc):
    return librosa.feature.mfcc(S=intrmd_melspec, sr=sr, n_mfcc=num_mfcc).reshape(-1, num_mfcc)

_mfcc_length = lambda obj: obj.num_mfcc
mfcc = FeatureSpec('mfcc', _mfcc, inputs=['intrmd_melspec'], save=True,
    length=_mfcc_length, dtype=np.float32, jagged_length=True, tags=['audio_representation', 'raw_audio'])


# def _rms(stft, win_length, hop_length):

#     return librosa.feature.rms(S=stft, frame_length=win_length, hop_length=hop_length).reshape(-1, 1)

# rms = FeatureSpec('rms', _rms, inputs=['stft'], save=True,
#     length=1, dtype=np.float32, jagged_length=True, tags=['raw_audio', 'audio_feature'])


# def _spectral_centroid(stft, num_fft, win_length, hop_length):
#     return librosa.feature.spectral_centroid(S=stft, 
#         n_fft=num_fft, win_length=win_length, hop_length=hop_length).reshape(-1, 1)

# spectral_centroid = FeatureSpec('spectral_centroid', _spectral_centroid, inputs=['stft'], save=True,
#     length=1, dtype=np.float32, jagged_length=True, tags=['raw_audio', 'audio_feature'])


def _raw_formant(wav, sr, win_length, hop_length, num_formants, formant_window_shape):
    if formant_window_shape == 'gaussian':
        window = scipy.signal.gaussian(win_length + 2, 0.45 * (win_length - 1) / 2)[1:win_length + 1]
    else:
        window = np.hanning(win_length + 2)[1:win_length + 1]
    wav = np.pad(wav, int(win_length // 2), mode='constant')
    frames = librosa.util.utils.frame(wav, frame_length=win_length, hop_length=hop_length)
    num_frames = frames.shape[1]
    freqs = np.zeros((num_frames, num_formants))
    bws = np.zeros((num_frames, num_formants))
    for i in range(num_frames):
        try:
            a = librosa.lpc(window * frames[:, i], order=num_formants * 2)
        except FloatingPointError:
            continue
        r = np.roots(a)
        # mask = np.sqrt(np.square(np.real(a)) + np.square(np.imag(a))) > 0.7
        mask = np.imag(r) > 0
        r = r[mask]
        freq = np.arctan2(np.imag(r), np.real(r)) * sr * (1./(2*np.pi))
        inds = np.argsort(freq)
        freqs[i, :inds.shape[0]] = freq[inds]
        bws[i, :inds.shape[0]] = (-1. / 2) * sr * (1. / (2 * np.pi)) * np.log(np.abs(r[inds]))
    return freqs, bws

raw_formant = FeatureSpec('raw_formant', _raw_formant, inputs=['wav'])


def _formant(raw_formant):
    return raw_formant[0]

_formant_length = lambda obj: obj.num_formants
formant = FeatureSpec('formant', _formant, inputs=['raw_formant'], 
    length=_formant_length, dtype=np.float32, jagged_length=True, never_store=True, save=True,
    normalization=MaskedMeanVarianceNormalization(0.0), tags=['raw_audio', 'audio_feature'])


def _formant_bw(raw_formant):
    return raw_formant[1]

formant_bw = FeatureSpec('formant_bw', _formant_bw, inputs=['raw_formant'], 
    length=_formant_length, dtype=np.float32, jagged_length=True, never_store=True, save=True,
    normalization=MaskedMeanVarianceNormalization(0.0), tags=['raw_audio', 'audio_feature'])


def _pitch_crepe(wav, sr, hop_length, crepe_model_capacity):
    _, frequency, _, _ = crepe.predict(wav, sr, 
        model_capacity=crepe_model_capacity, step_size=hop_length, viterbi=False, verbose=False)
    frequency[frequency > 0.0] = sr * 1. / frequency[frequency > 0.0]
    return frequency.reshape(-1)

pitch_crepe = FeatureSpec('pitch_crepe', _pitch_crepe, inputs=['wav'], save=True,
    length=1, dtype=np.float32, jagged_length=True, tags=['raw_audio', 'audio_feature'])


def _pitch_swipe(wav, sr, hop_length):
    return pysptk.sptk.swipe(wav.astype(np.float64), sr, hop_length, 
        min=50, max=600, otype='f0', threshold=0.3).reshape(-1)

pitch_swipe = FeatureSpec('pitch_swipe', _pitch_swipe, inputs=['wav'], save=True,
    length=1, dtype=np.float32, jagged_length=True, tags=['raw_audio', 'audio_feature'])


def _pitch_rapt(wav, sr, hop_length):
    return pysptk.sptk.rapt(wav.astype(np.float32), sr, hop_length, min=50, max=600, otype='f0').reshape(-1)

pitch_rapt = FeatureSpec('pitch_rapt', _pitch_rapt, inputs=['wav'], save=True,
    length=1, dtype=np.float32, jagged_length=True, tags=['raw_audio', 'audio_feature'])


def _zero_crossing_rate(wav, win_length, hop_length):
    return librosa.feature.zero_crossing_rate(wav, frame_length=win_length, hop_length=hop_length).reshape(-1)

zero_crossing_rate = FeatureSpec('zero_crossing_rate', _zero_crossing_rate, inputs=['wav'], save=True,
    length=1, dtype=np.float32, jagged_length=True, tags=['raw_audio', 'audio_feature'])


def _energy(wav, win_length, hop_length):
    wav = np.pad(wav, int(win_length // 2), mode='edge')
    wav = librosa.util.frame(wav, win_length, hop_length)
    wav = librosa.filters.get_window('hamming', win_length)[:, None] * wav
    return np.sum(np.power(wav, 2), axis=0)

energy = FeatureSpec('energy', _energy, inputs=['wav'], save=True,
    length=1, dtype=np.float32, jagged_length=True, tags=['raw_audio', 'audio_feature'])


def _magnitude(wav, win_length, hop_length):
    wav = np.pad(wav, int(win_length // 2), mode='edge')
    wav = librosa.util.frame(wav, win_length, hop_length)
    wav = librosa.filters.get_window('hamming', win_length)[:, None] * np.abs(wav)
    return np.sum(wav, axis=0)

magnitude = FeatureSpec('magnitude', _magnitude, inputs=['wav'], save=True,
    length=1, dtype=np.float32, jagged_length=True, tags=['raw_audio', 'audio_feature'])


class AudioArguments(Arguments):
    
    @property
    def parser(self):
        parser = super().parser
        parser.add_argument('--sr', type=int, default=16000)
        parser.add_argument('--win_t', type=float, default=0.025)
        parser.add_argument('--hop_t', type=float, default=0.010)
        parser.add_argument('--stft_window', type=str, default='hamming')
        parser.add_argument('--preemphasis', type=float, default=0.97)
        parser.add_argument('--num_fft', type=int, default=400)
        parser.add_argument('--num_mels', type=int, default=80)
        parser.add_argument('--num_mfcc', type=int, default=20)
        parser.add_argument('--norm_mel', type=bool, default=None)
        parser.add_argument('--log_mel', type=bool, default=True)
        parser.add_argument('--log_floor', type=int, default=-20)
        parser.add_argument('--crepe_model_capacity', type=str, default='small', \
            choices=['tiny', 'small', 'medium', 'large', 'full'])
        parser.add_argument('--num_formants', type=int, default=4)
        parser.add_argument('--formant_window_shape', type=str, default='gaussian')
        return parser

    def parse(self, args):
        super().parse(args)
        self.kwargs['win_length'] = int(self.kwargs['sr'] * self.kwargs['win_t'])
        self.kwargs['hop_length'] = int(self.kwargs['sr'] * self.kwargs['hop_t'])
