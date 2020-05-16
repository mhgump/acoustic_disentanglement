import tempfile
import numpy as np

import speech_representations.scripts.create_dataset as create_dataset
import speech_representations.parsers as parser_module
from speech_representations.features import list_features_by_tags
from speech_representations.data import DatasourceReader, SequenceEpoch, RandomSegmentEpoch


def get_parser_class(dataset_name):
    class_name = dataset_name.title() + 'Parser'
    assert hasattr(parser_module, class_name), 'No parser implemented for this dataset'
    parser_class = getattr(parser_module, class_name)
    return parser_class


def run():
    with tempfile.TemporaryDirectory() as directory:
        partition_name = 1
        seq_batch_size = 2
        seg_batch_size = 6
        window_length = 8
        K = 2
        num_mels = 60
        dataset_name = 'test'

        args, unknown = create_dataset.parse_args([
            directory,       # source_directory
            directory,       # target_directory
            dataset_name,    # dataset_name
            '--num_processes=1',
            '--feature_list=all_audio,speaker,phoneme',
            '--partition_list=0,1',
            '--split_size=1',
            '--num_mels={}'.format(num_mels),])
        create_dataset.run(args, unknown)

        feature_list = list_features_by_tags(tags=['raw_audio']) + ['phoneme']
        filename = DatasourceReader.get_filename(directory, dataset_name)
        parser_class = get_parser_class(dataset_name)
        arguments = parser_class.arguments_class()
        reader = DatasourceReader(filename, partition_name)
        arguments.decode(reader.arguments)
        parser = parser_class(args.source_directory, **arguments.kwargs)
        assert np.all(set(reader.valueset['speaker']) == set(parser.speaker.valueset))
        assert np.all(set(reader.valueset['phoneme']) == set(parser.phoneme.valueset))

        seq_epoch = SequenceEpoch(reader.ptn_num_items[partition_name], seq_batch_size)
        seg_epoch = RandomSegmentEpoch(seq_epoch, reader.get_windowing_length_func('melspec'), seg_batch_size, window_length, K)

        last_seq_batch = False
        for seq_indices in seq_epoch:
            _seq_batch_size = seq_batch_size
            if seq_indices.shape[0] != seq_batch_size:
                assert not last_seq_batch
                last_seq_batch = True
                _seq_batch_size = seq_indices.shape[0]

            speaker = reader.get_data('speaker', seq_indices)
            assert len(speaker.shape) == 2
            assert speaker.shape[0] == _seq_batch_size
            for feature_name in feature_list:
                padded_value, mask = reader.get_padded_batch(feature_name, seq_indices)
                assert padded_value.shape[:2] == mask.shape
                assert padded_value.shape[0] == _seq_batch_size
                assert len(padded_value.shape) == 3
                if feature_name == 'melspec':
                    assert padded_value.shape[2] == num_mels
        last_seg_batch = False
        for _seq_indices, start_pcts, length_pcts in seg_epoch:
            _seg_batch_size = seg_batch_size
            assert _seq_indices.shape[0] == start_pcts.shape[0] == length_pcts.shape[0]
            if _seq_indices.shape[0] != seg_batch_size:
                assert not last_seg_batch
                last_seg_batch = True
                _seg_batch_size = _seq_indices.shape[0]
            for feature_name in feature_list:
                value = reader.get_aligned_batch(feature_name, _seq_indices, start_pcts, length_pcts, window_length)
                assert len(value.shape) == 3
                assert value.shape[0] == _seg_batch_size
                assert value.shape[1] == window_length
                if feature_name == 'melspec':
                    assert value.shape[2] == num_mels


if __name__ == '__main__':
    run()
