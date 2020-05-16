import numpy as np

from speech_representations.features import transformer, register_feature, \
    FeatureSpec, Normalization, MaskedMeanVarianceNormalization, InputTransformer

input_array = np.ones(5,)
parameter_0 = 2.
parameter_1 = 3.
parameter_2 = 5.
parameter_3 = 7.

def _test_1(parameter_3):
    return parameter_3

def _test_5_0():
    return parameter_0

def _test_5(test_4, test_5_0, parameter_3):
    return test_5_0 * parameter_3 * test_4

test_1 = FeatureSpec('test_1', _test_1, parameters=['parameter_3'], normalize=True, normalization=lambda obj: obj.normalization, tags=['test1'])
test_5_0 = FeatureSpec('test_5_0', _test_5_0, normalize=True)
test_5 = FeatureSpec('test_5', _test_5, parameters=['parameter_3'], normalize=True, tags=['test2'])


@transformer(tags=['test1', 'test2'])
class TestTransformer:

    def __init__(self, input_array, parameter_1, parameter_2, parameter_3, normalization):
        self.input = input_array
        self.parameter_1 = parameter_1
        self.parameter_2 = parameter_2
        self.parameter_3 = parameter_3
        self.normalization = normalization

    @register_feature
    def test_2(self, test_1):
        return test_1 * self.parameter_1

    @register_feature(normalize=True, normalization=lambda obj: obj.normalization)
    def test_3(self, test_2):
        return test_2 * self.parameter_2

    @register_feature(normalize=True, normalization=lambda obj: obj.normalization)
    def test_4(self):
        return self.input

TestInputTransformer = transformer(tags=['test2'])(InputTransformer)

def run():
    normalization = MaskedMeanVarianceNormalization(0.0)
    test = TestTransformer(input_array, parameter_1, parameter_2, parameter_3, normalization)
    assert test.get('test_1') == parameter_3
    assert test.get('test_2') == parameter_1 * parameter_3
    assert test.get('test_3') == parameter_1 * parameter_2 * parameter_3
    assert np.all(test.get('test_4') == input_array)
    assert np.all(test.get('test_5') == parameter_0 * parameter_3 * input_array)
    assert isinstance(getattr(getattr(test, 'test_1'), 'normalization'), MaskedMeanVarianceNormalization)
    assert not hasattr(getattr(test, 'test_2'), 'normalization')
    assert isinstance(getattr(getattr(test, 'test_3'), 'normalization'), MaskedMeanVarianceNormalization)
    assert isinstance(getattr(getattr(test, 'test_4'), 'normalization'), MaskedMeanVarianceNormalization)
    assert isinstance(getattr(getattr(test, 'test_5'), 'normalization'), Normalization)

    input_test = TestInputTransformer(['test_4'], parameter_3=parameter_3)
    try:
        input_test.get('test_4')
        assert False, 'InputTransformer should have raised an exception'
    except LookupError:
        pass
    input_test.set('test_4', input_array)
    assert np.all(input_test.get('test_4') == input_array)
    assert np.all(input_test.get('test_5') == parameter_0 * parameter_3 * input_array)


if __name__ == '__main__':
    run()
