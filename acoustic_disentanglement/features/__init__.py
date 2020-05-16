from speech_representations.features.feature_registry import transformer, register_feature, FeatureSpec, list_features_by_tags
from speech_representations.features.normalization import *
from speech_representations.features.raw_audio import AudioArguments
import speech_representations.features.second_order
from speech_representations.features.reconstructions import ReconstructionTransformer


@transformer(tags=['raw_audio', 'second_order'])
class AudioTransformer:
    arguments_class = AudioArguments
    def __init__(self, **kwargs):
        for kwarg_name, kwarg in kwargs.items():
            setattr(self, kwarg_name, kwarg)
