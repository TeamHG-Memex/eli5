import pytest
import numpy as np

from eli5._feature_names import FeatureNames


# See also test_sklearn_utils.py::test_get_feature_names


def test_feature_names_filter_by_re():
    filtered, indices = (
        FeatureNames(['one', 'two', 'twenty-two']).filtered_by_re('two'))
    assert indices == [1, 2]
    assert list(filtered) == ['two', 'twenty-two']

    filtered, indices = (
        FeatureNames({1: 'two', 3: 'twenty-two', 5: 'two-thirds'}, unkn_template='%d',
                     n_features=6, bias_name='foo')
        .filtered_by_re('^two'))
    assert indices == [1, 5]
    assert filtered.bias_name == 'foo'
    assert filtered.unkn_template == '%d'
    assert list(filtered) == ['two', 'two-thirds', 'foo']

    filtered, indices = (
        FeatureNames(unkn_template='x%d', n_features=6).filtered_by_re('x'))
    assert indices == []


def test_init():
    with pytest.raises(ValueError):
        FeatureNames()
    with pytest.raises(ValueError):
        FeatureNames(unkn_template='%d')
    with pytest.raises(ValueError):
        FeatureNames(n_features=10)
    with pytest.raises(ValueError):
        FeatureNames(['a'], n_features=10)
    with pytest.raises(TypeError):
        FeatureNames({'a', 'b'})
    with pytest.raises(ValueError):
        FeatureNames({0: 'a', 1: 'b'}, n_features=10)
    FeatureNames(unkn_template='%d', n_features=10)
    FeatureNames(['a', 'b'])
    FeatureNames({0: 'a', 1: 'b'})
    FeatureNames({0: 'a', 1: 'b'}, n_features=10, unkn_template='x%d')


def test_slice():
    FN = FeatureNames
    assert FN(['one', 'two', 'three'])[1:] == ['two', 'three']
    assert FN(['one', 'two', 'three'])[:-2] == ['one']
    assert FN(['one', 'two', 'three'])[1:] == ['two', 'three']
    assert FN({1: 'one'}, n_features=3, unkn_template='x%d')[:] \
        == ['x0', 'one', 'x2']
    assert FN({1: 'one'}, n_features=3, unkn_template='x%d',
              bias_name='bias')[-3:] \
        == ['one', 'x2', 'bias']
    assert FN(['one', 'two', 'three'], bias_name='bias')[-1:] == ['bias']
    assert FN(np.array(['one', 'two', 'three']), bias_name='bias')[-1:] \
        == ['bias']
    assert FN(np.array(['one', 'two', 'three']), bias_name='bias')[-2:] \
        == ['three', 'bias']
    assert list(FN(np.array(['one', 'two', 'three']))[-2:]) == ['two', 'three']


@pytest.mark.parametrize(
    ['feature_names'], [
        [FeatureNames(['x1', 'x2', 'x3'])],
        [FeatureNames(['x1', 'x2', 'x3'], bias_name='<BIAS>')],
        [FeatureNames(np.array(['x1', 'x2', 'x3']))],
        [FeatureNames({0: 'x1', 1: 'x2'})],
        [FeatureNames(n_features=5, unkn_template='%d')],
    ])
def test_add_feature(feature_names):
    len_before = len(feature_names)
    storage = feature_names.feature_names
    new_feature = 'new'
    new_idx = feature_names.add_feature(new_feature)
    assert len(feature_names) == len_before + 1
    assert feature_names[new_idx] == new_feature
    if storage is not None:
        assert storage is not feature_names.feature_names
    # TODO - one more add
