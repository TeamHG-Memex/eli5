# Method for determining feature importances follows an idea from
# http://blog.datadive.net/interpreting-random-forests/.
# Implementations are in eli5.xgboost, eli5.lightgbm and
# eli5.sklearn.explain_prediction.
from eli5.base import Explanation, TargetExplanation
from eli5.sklearn.text import add_weighted_spans
from eli5.utils import get_target_display_names

DECISION_PATHS_DESCRIPTION = """
Feature weights are calculated by following decision paths in trees
of an ensemble. Each leaf has an output score, and expected scores can also be
assigned to parent nodes. Contribution of one feature on the decision path
is how much expected score changes from parent to child. Weights of all 
features sum to the output score of the estimator.
"""


DECISION_PATHS_CAVEATS = """
Caveats:
1. Feature weights just show if the feature contributed positively or
   negatively to the final score, and does show how increasing or decreasing
   the feature value will change the prediction.
2. In some cases, feature weight can be close to zero for an important feature.
   For example, in a single tree that computes XOR function, the feature at the
   top of the tree will have zero weight because expected scores for both
   branches are equal, so decision at the top feature does not change the
   expected score. For an ensemble predicting XOR functions it might not be
   a problem, but it is not reliable if most trees happen to choose the same
   feature at the top.
"""


DECISION_PATHS_EXPLANATION = "".join([
    DECISION_PATHS_DESCRIPTION,
    DECISION_PATHS_CAVEATS
])


DESCRIPTION_CLF_MULTICLASS = """
Features with largest coefficients per class.
""" + DECISION_PATHS_EXPLANATION

DESCRIPTION_CLF_BINARY = """
Features with largest coefficients.
""" + DECISION_PATHS_EXPLANATION

DESCRIPTION_REGRESSION = DESCRIPTION_CLF_BINARY


def get_decision_path_explanation(estimator, doc, vec, vectorized,
                                  original_display_names,
                                  target_names, targets, top_targets,
                                  is_regression, is_multiclass, proba,
                                  get_score_feature_weights):

    display_names = get_target_display_names(
        original_display_names, target_names, targets, top_targets, proba)

    explanation = Explanation(
        estimator=repr(estimator),
        method='decision paths',
        description={
            (False, False): DESCRIPTION_CLF_BINARY,
            (False, True): DESCRIPTION_CLF_MULTICLASS,
            (True, False): DESCRIPTION_REGRESSION,
        }[is_regression, is_multiclass],
        is_regression=is_regression,
        targets=[],
    )

    if is_multiclass:
        for label_id, label in display_names:
            score, feature_weights = get_score_feature_weights(label_id)
            target_expl = TargetExplanation(
                target=label,
                feature_weights=feature_weights,
                score=score,
                proba=proba[label_id] if proba is not None else None,
            )
            add_weighted_spans(doc, vec, vectorized, target_expl)
            explanation.targets.append(target_expl)
    else:
        score, feature_weights = get_score_feature_weights(0)
        target_expl = TargetExplanation(
            target=display_names[-1][1],
            feature_weights=feature_weights,
            score=score,
            proba=proba[1] if proba is not None else None,
        )
        add_weighted_spans(doc, vec, vectorized, target_expl)
        explanation.targets.append(target_expl)

    return explanation
