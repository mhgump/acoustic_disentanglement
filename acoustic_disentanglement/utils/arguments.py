""" utils/arguments.py

Arguments class allows argument handlers to be extended and passed around neatly
Argument handlers take in list of arguments (like argparse) and can produce key-value dictionaries
and encode/decode themselves for saving/loading. YML is used to specify configs that enumerate
multiple argument sets that can then be used to instantiate multiple models/datasets/jobs/etc.
"""
import yaml
from itertools import product
import argparse
import os
import json

def setting_product(settings_1, settings_2):
    settings = []
    for setting_1, setting_2 in product(settings_1, settings_2):
        settings += [[*setting_1, *setting_2]]
    return settings


def format_as_command(key, value):
    if value == 'True' or (type(value) is bool and value):
        # True values should be formatted as tags
        return '--{}'.format(key)
    elif value == 'False' or (type(value) is bool and not value):
        # False values should be ignored
        return None
    elif value is not None:
        # Otherwise arguments should be in key-value format
        return '--{}={}'.format(key, value)


def _get_all_argument_combinations(config, init_args=None):
    init_args = init_args or []
    settings = [init_args]
    for key, value in config.items():
        if type(value) is list:
            new_settings = []
            for sub_config in value:
                sub_settings = _get_all_argument_combinations(sub_config)
                new_settings += setting_product(settings, sub_settings)
            settings = new_settings
        else:
            for setting in settings:
                if key not in setting:
                    formatted = format_as_command(key, value)
                    if formatted is not None:
                        setting += [formatted]
    return settings


def get_all_argument_combinations(yaml_filepath, global_arguments): 
    """
    :str yaml_filepath: path to yaml formatted file specifying parameter settings
    list values indicate that disctinct models with different settings
    :str global_arguments: unparsed arguments that all options should use
    """
    config = yaml.load(open(yaml_filepath), Loader=yaml.BaseLoader)
    return _get_all_argument_combinations(config, global_arguments)


class Arguments:
    
    def __init__(self, kwargs=None):
        self.kwargs = kwargs

    @property
    def parser(self):
        return argparse.ArgumentParser()

    def parse(self, args, return_unknown=False):
        args_namespace, unknown = self.parser.parse_known_args(args)
        self.kwargs = { k: v for k, v in args_namespace._get_kwargs() }
        if return_unknown:
            return unknown

    def encode(self):
        return json.dumps(self.kwargs)

    def decode(self, encoding):
        self.kwargs = json.loads(encoding)

    def format_parseable(self):
        parseable = [format_as_command(key, value) for key, value in self.kwargs.items()]
        return [e for e in parseable if e is not None]

