

class SubscriptableDescriptor:

    def __init__(self, *args):
        self.prefixes = list(map(str, args))

    def get_key(self, *keys):
        keys = list(map(str, keys))
        key = '/'.join([*self.prefixes, *keys])
        return key

    def _contains(self, key):
        raise NotImplementedError

    def _get(self, key):
        raise NotImplementedError

    def _set(self, key, value):
        raise NotImplementedError

    def _del(self, key):
        raise NotImplementedError

    def __contains__(self, keys):
        if not isinstance(keys, tuple):
            keys = (keys,)
        keys = self.get_key(*keys)
        return self._contains(keys)

    def __getitem__(self, keys):
        if not isinstance(keys, tuple):
            keys = (keys,)
        keys = self.get_key(*keys)
        if self._contains(keys):
            return self._get(keys)
        else:
            raise AttributeError('No property \'{}\''.format(keys))

    def __setitem__(self, keys, value):
        if not isinstance(keys, tuple):
            keys = (keys,)
        keys = self.get_key(*keys)
        return self._set(keys, value)

    def __delitem__(self, keys):
        if not isinstance(keys, tuple):
            keys = (keys,)
        keys = self.get_key(*keys)
        return self._del(keys)
