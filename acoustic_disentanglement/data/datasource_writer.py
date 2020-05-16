import h5py
import os
import glob
import numpy as np
import pandas as pd
import itertools
from time import time
from collections import defaultdict
import argparse
import shutil

from speech_representations.features import list_features_by_tags
from speech_representations.data.datasource import DatasourceBase, SplitDatasourceBase, bind_datasource_attributes
import speech_representations.parsers as parser_module

INTTYPE = np.int32


def get_parser_class(dataset_name):
    class_name = dataset_name.title() + 'Parser'
    assert hasattr(parser_module, class_name), 'No parser implemented for this dataset, {}'.format(class_name)
    parser_class = getattr(parser_module, class_name)
    return parser_class


def parse_feature_list(raw_feature_list):
    feature_list = raw_feature_list.split(',') if raw_feature_list is not None else []
    if 'all_audio' in feature_list:
        feature_list.remove('all_audio')
        feature_list += list_features_by_tags(tags=['raw_audio'])
    return feature_list


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('source_directory', type=str, help='Directory holding source data')
    parser.add_argument('target_directory', type=str, help='Directory where datafiles should be written')
    parser.add_argument('dataset_name', type=str, help='Name of dataset, must match parser implementation')
    subparsers = parser.add_subparsers(help='sub-command help')

    parser_setup = subparsers.add_parser('setup_job', help='Subcommand for intial data task')
    parser_setup.set_defaults(command=setup_job)
    parser_setup.add_argument('--split_size', type=int, help='Maximum size (in MB) per data split', required=False)
    parser_setup.add_argument('--partition_list', type=str,
        help='Partitions to be processed into split files')
    parser_setup.add_argument('--overwrite_file', action='store_true',
        help='When flag is included file will be constructed from scratch')

    parser_split = subparsers.add_parser('start_split_process', help='Subcommand for parallelizable split data tasks')
    parser_split.set_defaults(command=start_split_process)
    parser_split.add_argument('--num_processes', type=int, required=True, \
        help='Total number of processes, required to split data correctly.')
    parser_split.add_argument('--process_index', type=int,  required=True, \
        help='This process\' assigned index, 0 <= index < num_processes')
    parser_split.add_argument('--feature_list', type=str, \
        help='Features to be processed into split files')
    parser_split.add_argument('--keep_existing', action='store_false',
        help='When flag is included existing feature values will not be overwritten.')
    parser_split.add_argument('--max_processed', type=int, required=False,
        help='Limits the total number of processed audio files when provided.')

    parser_finalize = subparsers.add_parser('finalize_job', help='Subcommand for finalizing data task')
    parser_finalize.set_defaults(command=finalize_job)
    parser_finalize.add_argument('--feature_list', type=str, \
        help='Features to be processed into split files')

    if args is not None:
        return parser.parse_known_args(args)
    else:
        return parser.parse_known_args()

class DatasourceSplitWriter(DatasourceBase):

    datavalues = ['data', 'num_segments', 'segment_ends', 'sequence_length']
    grouped_attributes = ['dtype', 'length', 'jagged_length', 'variable_length_segments']
    attributes = ['source_directory', 'partition_name', 'start_index', 'end_index', 'num_items', 'feature_list']

    def check(self, feature_name):
        assert feature_name in self.data, 'Data value must be present'
        assert self.dtype[feature_name] is not None, 'Feature attributes must be set' 
        assert self.length[feature_name] is not None, 'Feature attributes must be set' 
        assert self.jagged_length[feature_name] is not None, 'Feature attributes must be set' 
        assert self.variable_length_segments[feature_name] is not None, 'Feature attributes must be set' 
        if self.jagged_length[feature_name]:
            assert feature_name in self.num_segments, 'Jagged length datatypes must have segment counts'
        if self.variable_length_segments[feature_name]:
            assert feature_name in self.segment_ends, 'Datatypes with variable length segments must have segment ends'
            assert feature_name in self.sequence_length, \
                'Datatypes with variable length segments must have sequence lengths'

    def initalize(self, source_directory, partition_name, start_index, end_index):
        self.source_directory = source_directory
        self.partition_name = partition_name
        self.start_index = start_index
        self.end_index = end_index
        self.num_items = end_index - start_index
        self.feature_list = []

    def add_feature(self, feature_name, dtype, length, jagged_length, variable_length_segments):
        self.dtype[feature_name] = np.sctype2char(dtype)
        self.length[feature_name] = length
        self.jagged_length[feature_name] = jagged_length
        self.variable_length_segments[feature_name] = variable_length_segments
        self.feature_list = list(self.feature_list) + [feature_name]
        self.add_dataset(self.data.get_key(feature_name), 
                          (self.num_items, length,), 
                          h5py.special_dtype(vlen=dtype) if jagged_length else dtype)
        if jagged_length:
            self.add_dataset(self.num_segments.get_key(feature_name),
                              (self.num_items, 1,),
                              INTTYPE)
        if variable_length_segments:
            self.add_dataset(self.segment_ends.get_key(feature_name),
                              (self.num_items, 1,),
                              h5py.special_dtype(vlen=INTTYPE))
            self.add_dataset(self.sequence_length.get_key(feature_name),
                              (self.num_items, 1,),
                              INTTYPE)

    def delete_feature(self, feature_name):
        feature_list = list(self.feature_list)
        feature_list.remove(feature_name)
        self.feature_list = feature_list
        del self.data[feature_name]
        if self.jagged_length[feature_name]:
            del self.num_segments[feature_name]
        if self.variable_length_segments[feature_name]:
            del self.segment_ends[feature_name]
            del self.sequence_length[feature_name]
        del self.length[feature_name]
        del self.jagged_length[feature_name]
        del self.variable_length_segments[feature_name]

    def add_dataset(self, path, shape, dtype):
        if path in self.file:
            del self.file[path]
        self.file.create_dataset(path, shape, dtype=dtype)

bind_datasource_attributes(DatasourceSplitWriter)

class SplitDatasourceWriter(SplitDatasourceBase):

    def __init__(self, filename):
        super().__init__(filename)

    def initialize_split_file(self, partition_name, start_index, end_index, split_index):
        split_filename = self.split_filename(split_index)
        split_file = DatasourceSplitWriter(split_filename)
        split_file.open()
        if split_file.partition_name is None:
            split_file.initalize(self.source_directory, partition_name, start_index, end_index)
        elif split_file.partition_name != partition_name \
                or split_file.start_index != start_index \
                or split_file.end_index != end_index:
            print('Overwriting old split file ({}, {}, {}) since ({}, {}, {}) must be written'.format(
                split_file.partition_name, split_file.start_index, split_file.end_index,
                partition_name, start_index, end_index))
            split_file.close()
            os.remove(split_filename)
            split_file = DatasourceSplitWriter(split_filename)
            split_file.open()
            split_file.initalize(self.source_directory, partition_name, start_index, end_index)
        split_file.close()
        self.partition_name[split_index] = partition_name
        self.start_index[split_index] = start_index
        self.end_index[split_index] = end_index

    def get_partition_feature_list(self, partition_name):
        # Get list of features both requested and present in this partition
        feature_set = None
        for split_index in self.split_indices[partition_name]:
            split_filename = self.split_filename(split_index)
            split_file = DatasourceSplitWriter(split_filename)
            split_file.open()
            split_feature_set = set(split_file.feature_list)
            if feature_set is None:
                feature_set = split_feature_set
            else:
                assert feature_set == split_feature_set, \
                    'All a partition\'s split files should contain the same features'
            split_file.close()
        return list(feature_set)

    def write_normalization(self, partition_name, feature_name, feature_spec, batch_size=100):
        num_items = self.ptn_num_items[partition_name]
        length = self.length[feature_name]
        normalization = feature_spec.normalization
        for start_ind in range(0, num_items, batch_size):
            end_ind = start_ind + batch_size
            data = self.data[partition_name, feature_name][start_ind:end_ind, :]
            data = np.concatenate(data.transpose(1,0).flatten()).reshape(data.shape[1], -1).transpose(1, 0)
            normalization.add(data)
        self.normalization[partition_name, feature_name] = normalization.save()

    def create_dataset(self, key, layout):
        if key in self.file:
            del self.file[key]
        self.file.create_virtual_dataset(key, layout)


def setup_job(args, unknown_args):
    parser_class = get_parser_class(args.dataset_name)
    arguments = parser_class.arguments_class()
    arguments.parse(unknown_args)
    parser = parser_class(args.source_directory, **arguments.kwargs)
    filename = SplitDatasourceWriter.get_filename(args.target_directory, args.dataset_name)
    partition_list = args.partition_list.split(',') if args.partition_list is not None else parser.partitions

    if args.overwrite_file:
        directory = SplitDatasourceWriter.get_directory(args.target_directory, args.dataset_name)
        if os.path.exists(directory):
            shutil.rmtree(directory)
    print('Dataset created or found at {}'.format(filename))
    dataset_writer = SplitDatasourceWriter(filename)
    dataset_writer.open()
    dataset_writer.source_directory = args.source_directory
    dataset_writer.data_size = 0
    dataset_writer.num_items = 0
    split_index = 0
    if partition_list is None:
        partition_list = parser.partitions
    else:
        partition_list = list(set(partition_list).intersection(set(parser.partitions)))
    dataset_writer.partition_names = sorted(partition_list)
    for partition_name in dataset_writer.partition_names:
        data_size_m = parser.get_partition_size(partition_name)
        data_size_f = parser.num_items(partition_name)
        split_size_f = int(data_size_f*args.split_size*(1./data_size_m)) if args.split_size is not None else data_size_f
        split_ind_start = split_index
        
        end_index = 0
        for start_index in range(0, data_size_f, split_size_f):
            end_index = min(data_size_f, start_index + split_size_f)
            dataset_writer.initialize_split_file(partition_name, start_index, end_index, split_index)
            split_index += 1
        if end_index != data_size_f:
            # TODO: Resize the splits so that the overhang chunk is not too small
            # This should have *very* minimal impact on performance.
            dataset_writer.initialize_split_file(partition_name, end_index, data_size_f, split_index)
            split_index += 1

        dataset_writer.ptn_data_size[partition_name] = data_size_m
        dataset_writer.ptn_num_items[partition_name] = data_size_f
        dataset_writer.split_indices[partition_name] = range(split_ind_start, split_index)
        dataset_writer.data_size += data_size_m
        dataset_writer.num_items += data_size_f
    dataset_writer.num_splits = split_index
    dataset_writer.arguments = arguments.encode()
    dataset_writer.close()
    print('Initialized job data file')


def start_split_process(args, unknown_args):
    parser_class = get_parser_class(args.dataset_name)
    filename = SplitDatasourceWriter.get_filename(args.target_directory, args.dataset_name)
    
    datasource_reader = SplitDatasourceBase(filename)
    datasource_reader.open(mode='r')
    arguments = parser_class.arguments_class()
    arguments.decode(datasource_reader.arguments)
    parser = parser_class(args.source_directory, **arguments.kwargs)
    num_splits = datasource_reader.num_splits
    datasource_reader.close()

    feature_list = parse_feature_list(args.feature_list)
    for split_index in range(args.process_index, num_splits, args.num_processes):
        print('Starting split file {}'.format(split_index))
        split_filename = SplitDatasourceBase.get_split_filename(args.target_directory, args.dataset_name, split_index)
        dataset_split = DatasourceSplitWriter(split_filename)
        dataset_split.open()
        assert dataset_split.partition_name in parser.partitions, 'Specified partition must be parsable'
        assert dataset_split.source_directory == parser.source_directory, 'Source directories must match'         
        feature_list = list(set(feature_list).intersection(parser.list_features(dataset_split.partition_name)))
        if not args.keep_existing:
            for feature_name in list(dataset_split.feature_list):
                dataset_split.delete_feature(feature_name)

        missing_features = set(feature_list).difference(set(dataset_split.feature_list))
        for feature_name in missing_features:
            dataset_split.add_feature(feature_name, 
                              parser.dtype(feature_name),
                              parser.length(feature_name),
                              parser.jagged_length(feature_name),
                              parser.variable_length_segments(feature_name))
        data_generator = parser.generate_data(dataset_split.partition_name, dataset_split.start_index, dataset_split.end_index)
        start_time = time()
        for index, _ in enumerate(data_generator):
            if args.max_processed is not None and args.max_processed < index:
                break
            for feature_name in missing_features:
                dtype = parser.dtype(feature_name)
                length = parser.length(feature_name)
                value = parser.get(feature_name)                
                if dataset_split.jagged_length[feature_name]:
                    if dataset_split.variable_length_segments[feature_name]:
                        value, ends = value
                        ends =  ends.reshape(-1, length).transpose().copy().astype(INTTYPE)
                        dataset_split.sequence_length[feature_name][index] = ends[0, -1]
                        dataset_split.segment_ends[feature_name][index] = ends
                    value = value.reshape(-1, length).transpose().copy().astype(dtype)
                    dataset_split.num_segments[feature_name][index] = value.shape[1]
                    dataset_split.data[feature_name][index] = value
                else:
                    dataset_split.data[feature_name][index, :] = np.array(value).astype(dtype)
        processing_time = time() - start_time
        print('Finished split file in {} seconds'.format(processing_time))
        dataset_split.close()


def finalize_job(args, unknown_args):
    filename = SplitDatasourceWriter.get_filename(args.target_directory, args.dataset_name)
    dataset_writer = SplitDatasourceWriter(filename)
    dataset_writer.open()
    try:
        _finalize_job(dataset_writer, args, unknown_args)
    finally:
        dataset_writer.close()

def _finalize_job(dataset_writer, args, unknown_args):
    parser_class = get_parser_class(args.dataset_name)
    feature_list = parse_feature_list(args.feature_list)

    arguments = parser_class.arguments_class()
    arguments.decode(dataset_writer.arguments)
    parser = parser_class(args.source_directory, **arguments.kwargs)

    # Setup each partition/feature
    for partition_name in dataset_writer.partition_names:
        num_items = dataset_writer.ptn_num_items[partition_name] 
        dataset_writer.feature_list[partition_name] = \
            dataset_writer.get_partition_feature_list(partition_name)
        requested_features = set(parser.list_features(partition_name)).intersection(feature_list)
        assert requested_features.issubset(dataset_writer.feature_list[partition_name]), \
            'Partition must contain all requested features'

        # Create data templates for each feature in this partition
        for feature_name in dataset_writer.feature_list[partition_name]:
            dataset_writer.dtype[feature_name] = np.sctype2char(parser.dtype(feature_name))
            dataset_writer.length[feature_name] = parser.length(feature_name)
            dataset_writer.jagged_length[feature_name] = parser.jagged_length(feature_name)
            dataset_writer.variable_length_segments[feature_name] = parser.variable_length_segments(feature_name)
            if getattr(parser, feature_name).valueset is not None:
                dataset_writer.valueset[feature_name] = getattr(parser, feature_name).valueset

            raw_dtype = parser.dtype(feature_name)
            dtype = h5py.special_dtype(vlen=raw_dtype) if dataset_writer.jagged_length[feature_name] else raw_dtype
            data_layout = h5py.VirtualLayout(shape=(num_items, dataset_writer.length[feature_name],), dtype=dtype)
            if dataset_writer.variable_length_segments[feature_name]:
                segment_ends_layout = \
                    h5py.VirtualLayout(shape=(num_items, 1,), dtype=h5py.special_dtype(vlen=INTTYPE))
                sequence_length_layout = h5py.VirtualLayout(shape=(num_items, 1,), dtype=INTTYPE)
            if dataset_writer.jagged_length[feature_name]:
                num_segment_layout = h5py.VirtualLayout(shape=(num_items, 1,), dtype=INTTYPE)

            for split_index in dataset_writer.split_indices[partition_name]:
                start_index = dataset_writer.start_index[split_index]
                end_index = dataset_writer.end_index[split_index]
                split_filename = dataset_writer.split_filename(split_index)
                split_file = DatasourceSplitWriter(split_filename)
                split_file.open()

                data_layout[start_index:end_index, :] = h5py.VirtualSource(split_filename,
                   name=split_file.data.get_key(feature_name),
                   shape=(end_index - start_index, dataset_writer.length[feature_name]), 
                   dtype=dtype)
                if dataset_writer.variable_length_segments[feature_name]:
                    segment_ends_layout[start_index:end_index, :] =  h5py.VirtualSource(split_filename,
                       name=split_file.segment_ends.get_key(feature_name),
                       shape=(end_index - start_index, 1),
                       dtype=INTTYPE)
                    sequence_length_layout[start_index:end_index, :] = h5py.VirtualSource(split_filename,
                       name=split_file.sequence_length.get_key(feature_name),
                       shape=(end_index - start_index, 1), 
                       dtype=INTTYPE)
                if dataset_writer.jagged_length[feature_name]:
                    num_segment_layout[start_index:end_index, :] = h5py.VirtualSource(split_filename,
                       name=split_file.num_segments.get_key(feature_name),
                       shape=(end_index - start_index, 1), 
                       dtype=INTTYPE)
                split_file.close()

            dataset_writer.create_dataset(dataset_writer.data.get_key(partition_name, feature_name), data_layout)
            if dataset_writer.variable_length_segments[feature_name]:
                dataset_writer.create_dataset(dataset_writer.segment_ends.get_key(partition_name, feature_name),
                    segment_ends_layout)
                dataset_writer.create_dataset(dataset_writer.sequence_length.get_key(partition_name, feature_name),
                    sequence_length_layout)
            if dataset_writer.jagged_length[feature_name]:
                dataset_writer.create_dataset(dataset_writer.num_segments.get_key(partition_name, feature_name),
                    num_segment_layout)
    print('Finished job data file', flush=True)

    # Add normalization of each feature that requests it
    for partition_name in dataset_writer.partition_names:
        for feature_name in dataset_writer.feature_list[partition_name]:
            feature_spec = getattr(parser, feature_name)
            if not feature_spec.normalize:
                continue
            dataset_writer.write_normalization(partition_name, feature_name, feature_spec)
    print('Finished saving normalization staistics for each feature')

if __name__ == '__main__':
    args, unknown = parse_args()
    args.command(args, unknown)
