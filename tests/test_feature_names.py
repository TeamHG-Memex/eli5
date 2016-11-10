import pytest

from eli5._feature_names import FeatureNames


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
    with pytest.raises(TypeError):
        FeatureNames({'a', 'b'})
    FeatureNames(unkn_template='%d', n_features=10)
    FeatureNames(['a', 'b'])
    FeatureNames({0: 'a', 1: 'b'})


# See also test_sklearn_utils.py::test_get_feature_names
