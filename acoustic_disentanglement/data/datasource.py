import h5py
import os

from speech_representations.utils.descriptors import SubscriptableDescriptor


class H5PyAttributeDescriptor:

    def __init__(self, name):
        self.name = name

    def __contains__(self, obj, type=None):
        return self.name in obj.file.attrs

    def __get__(self, obj, type=None):
        if self.name in obj.file.attrs:
            return obj.file.attrs[self.name]
        else:
            return None

    def __set__(self, obj, value):
        obj.file.attrs[self.name] = value


class H5PySubscriptableValueDescriptor(SubscriptableDescriptor):

    def __init__(self, obj, *args):
        self.obj = obj
        super().__init__(*args)

    def _contains(self, key):
        return key in self.obj.file

    def _get(self, key):
        return self.obj.file[key]

    def _del(self, key):
        del self.obj.file[key]


class H5PySubscriptableAttributeDescriptor(SubscriptableDescriptor):

    def __init__(self, obj, *args):
        self.obj = obj
        super().__init__(*args)

    def _contains(self, key):
        return key in self.obj.file.attrs

    def _get(self, key):
        return self.obj.file.attrs[key]

    def _set(self, key, value):
        self.obj.file.attrs[key] = value

    def _del(self, key):
        del self.obj.file.attrs[key]


class DatasourceBase:
    datavalues = []
    grouped_attributes = []
    attributes = []

    def __init__(self, filename):
        self.filename = filename
        self.file = None
        for attribute_name in self.__class__.datavalues:
            setattr(self, attribute_name, H5PySubscriptableValueDescriptor(self, attribute_name))
        for attribute_name in self.__class__.grouped_attributes:
            setattr(self, attribute_name, H5PySubscriptableAttributeDescriptor(self, attribute_name))

    def open(self, mode='a'):
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        self.file = h5py.File(self.filename, mode)

    def close(self):
        if self.file is not None:
            self.file.close()


def bind_datasource_attributes(_class):
    assert issubclass(_class, DatasourceBase), '{} is not a Datasource'.format(_class)
    for attribute_name in _class.attributes:
        setattr(_class, attribute_name, H5PyAttributeDescriptor(attribute_name))


class SplitDatasourceBase(DatasourceBase):
    
    datavalues = ['data', 'num_segments', 'segment_ends', 'sequence_length',]
    grouped_attributes = ['partition_name', 'start_index', 'end_index', # Accesible by split
        'ptn_data_size', 'ptn_num_items', 'split_indices', 'feature_list', # Accesible by partition
        'valueset', 'dtype', 'length', 'jagged_length', 'variable_length_segments',  # Accesible by feature
        'normalization',] # Accesible by partition, feature tuple
    attributes = ['source_directory', 'data_size', 'num_items', 'partition_names', 'num_splits', 'arguments']

    @classmethod
    def get_directory(cls, target_directory, dataset_name):
        return os.path.join(target_directory, dataset_name)

    @classmethod
    def get_filename(cls, target_directory, dataset_name):
        return os.path.join(cls.get_directory(target_directory, dataset_name), 'data.hdf5')
    
    @classmethod
    def get_split_directory(cls, target_directory, dataset_name):
        return os.path.join(cls.get_directory(target_directory, dataset_name), 'split_files')

    @classmethod
    def get_split_filename(cls, target_directory, dataset_name, split_index):
        return os.path.join(cls.get_split_directory(target_directory, dataset_name), 'split_{}.hdf5'.format(split_index))

    def split_filename(self, split_index):
        return os.path.join(os.path.dirname(self.filename), 'split_files', 'split_{}.hdf5'.format(split_index))

bind_datasource_attributes(SplitDatasourceBase)
