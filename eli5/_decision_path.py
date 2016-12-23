# Method for determining feature importances follows an idea from
# http://blog.datadive.net/interpreting-random-forests/.
# Implementations are in eli5.xgboost and eli5.sklearn.explain_prediction


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
