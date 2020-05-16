""" Feature registry

Decorators that allow easier feature definition. Features are registered as feature specifications with FeatureSpec.
Features can be applied by the class wrapper 'transformer' to be used by an object.
The result is that features can be defined abstractly once but used concretely in multiple places.
"""
from collections import defaultdict
import inspect
from abc import ABCMeta, abstractmethod
import types
import inspect
import argparse
import json

from speech_representations.features.normalization import *


class FeatureRegistry:
    _FEATURES = {}
    _TAG_GROUPS = defaultdict(list)

    @classmethod
    def register(cls, feature_spec, tags):
        tags = tags or []
        cls._FEATURES[feature_spec.feature_name] = feature_spec
        for tag_name in tags:
            cls._TAG_GROUPS[tag_name] += [feature_spec.feature_name]

    @classmethod
    def get_spec(cls, feature_name):
        return cls._FEATURES[feature_name]

    @classmethod
    def list_dependencies(cls, feature_name):
        dependencies = set()
        for feature in cls._FEATURES[feature_name].inputs:
            if feature in cls._FEATURES:
                dependencies.add(feature)
                dependencies.update(cls.list_dependencies(feature))
        return list(dependencies)

    @classmethod
    def list_features_by_tags(cls, tags=None, include_dependencies=False):
        tags = tags or []
        features = set()
        for tag_name in tags:
            features.update(cls._TAG_GROUPS[tag_name])
        if include_dependencies:
            for feature in list(features):
                features.update(cls.list_dependencies(feature))
        return list(features)

list_features_by_tags = FeatureRegistry.list_features_by_tags


class FeatureSpec():

    SUPPORTED_LATENT_TYPES = ['point', 'sequential_point', 'sequential_gaussian']

    def __init__(self, feature_name, method, uses_object=False, never_store=False, tags=None, inputs=None, 
            parameters=None, **kwargs):
        self.feature_name = feature_name
        self.method = method
        self.kwargs = kwargs
        self.uses_object = uses_object
        self.never_store = never_store
        if self.uses_object:
            self.inputs = inspect.getfullargspec(self.method).args[1:]
            self.parameters = []
        else:
            tags = tags or []
            FeatureRegistry.register(self, tags)
            if inputs is None:
                self.parameters = parameters or []
                self.inputs = list(set(inspect.getfullargspec(self.method).args).difference(self.parameters))
            else:
                self.inputs = inputs
                self.parameters = list(set(inspect.getfullargspec(self.method).args).difference(self.inputs))

    def generate(self, obj):
        kwargs = { kwarg_name:
            kwarg if not isinstance(kwarg, types.FunctionType) else kwarg(obj)
            for kwarg_name, kwarg in self.kwargs.items() }
        for kwarg_name, kwarg in self.kwargs.items():
            if isinstance(kwarg, types.FunctionType):
                kwarg = kwarg(obj)
            kwargs[kwarg_name] = kwarg
        self.init_normalized_feature(**kwargs)
        self.init_saveable_feature(**kwargs)

    def get_parameters(self, obj):
        parameter_values = {}
        for parameter in self.parameters:
            parameter_values[parameter] = getattr(obj, parameter)
        return parameter_values

    def init_normalized_feature(self, normalize=True, normalization=None, **kwargs):
        self.normalize = normalize
        if self.normalize:
            self.normalization = normalization or MeanVarianceNormalization()
            assert isinstance(self.normalization, Normalization)

    def init_saveable_feature(self, save=False, length=None, dtype=None, jagged_length=None, variable_length_segments=None,
            valueset=None, **kwargs):
        self.save = save
        if self.save:
            self.length = length or 1
            self.dtype = dtype or np.int32
            self.jagged_length = jagged_length or False
            self.variable_length_segments = variable_length_segments or False
            self.valueset = valueset


def register_feature(_registering_method=None, **kwargs):
    def _func(_method):
        feature_spec = FeatureSpec(_method.__name__, _method, uses_object=True, **kwargs)
        return feature_spec
    if _registering_method is None:
        return _func
    else:
        return _func(_registering_method)


def _attach_feature_spec(transformer, feature_spec, in_place=True):
    feature_name = feature_spec.feature_name
    feature_spec.generate(transformer)
    if not feature_spec.uses_object:
        transformer._registered_feature_parameters[feature_name] = feature_spec.get_parameters(transformer)
    if not in_place:
        setattr(transformer, feature_name, feature_spec)


def generate_transformer_init(cls):
    base_init = cls.__init__
    def _transformer_init(transformer, *args, **kwargs):
        base_init(transformer, *args, **kwargs)
        transformer._registered_feature_values = {}
        transformer._registered_feature_parameters = {}
        for feature_name in transformer._attached_feature_list:
            feature_spec = getattr(transformer, feature_name)
            _attach_feature_spec(transformer, feature_spec, in_place=False)
        registered_method_list = list()
        for attr_name in dir(transformer):
            attr = getattr(transformer, attr_name)
            if isinstance(attr, FeatureSpec) and attr.uses_object:
                _attach_feature_spec(transformer, attr, in_place=True)
                registered_method_list += [attr_name]
        transformer._registered_method_list = registered_method_list
        transformer._registered_feature_list = transformer._registered_method_list + transformer._attached_feature_list
    return _transformer_init


def _transformer_get(transformer, feature_name, retain_graph=True):
    if feature_name in transformer._registered_feature_values:
        return transformer._registered_feature_values[feature_name]
    assert hasattr(transformer, feature_name), 'No such feature {}'.format(feature_name)
    feature_spec = getattr(transformer, feature_name)
    inputs = { input_name: transformer.get(input_name, retain_graph) for input_name in feature_spec.inputs }
    if feature_spec.uses_object:
        value = feature_spec.method(transformer, **inputs)
    else:
        value = feature_spec.method(**inputs, **transformer._registered_feature_parameters[feature_name])
    if retain_graph and not feature_spec.never_store:
        transformer._registered_feature_values[feature_name] = value
    return value


def _transformer_set(transformer, feed_dict):
    transformer._registered_feature_values = {}
    for feature_name, value in feed_dict.items():
        transformer._registered_feature_values[feature_name] = value


def _transformer(_class, tags=None):
    tags = tags or []
    _class._attached_feature_list = FeatureRegistry.list_features_by_tags(tags, include_dependencies=True)
    _class.__init__ = generate_transformer_init(_class)
    _class.get = _transformer_get
    _class.set = _transformer_set
    for feature_name in _class._attached_feature_list:
        setattr(_class, feature_name, FeatureRegistry.get_spec(feature_name))
    return _class


def transformer(_class=None, tags=None):
    if _class is None:
        return lambda _class: _transformer(_class, tags)
    else:
        return _transformer(_class, tags)
