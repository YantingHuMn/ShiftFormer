"""ShiftFormer: transformation x normalization selection with Transformer denoise (v1).

    import shiftformer
    result = shiftformer.choose_combination(train, test)   # raw matrix, dropout matrix
    result.summary; result.best_by_pearson; result.best_by_spearman; result.report
    result.save("out_dir")
"""
from .pipeline import (ChooseCombinationResult, StopForUser, choose_combination,  # noqa: F401
                       load_result, merge_results)
from .preprocess import NORMALIZATIONS, TRANSFORMATIONS, all_combos  # noqa: F401

__version__ = "0.2.0"
