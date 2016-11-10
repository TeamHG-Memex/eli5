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
