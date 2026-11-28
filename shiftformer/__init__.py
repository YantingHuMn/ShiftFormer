"""ShiftFormer: Transformer denoise with log2(count+2) and a rule-chosen normalization.

Default workflow:
    import shiftformer
    res = shiftformer.run(counts)            # raw count matrix (cells x genes)
    res.reconstructions["pearson"]; res.reconstructions["spearman"]; res.config; res.report
    res.save("out_dir")                      # {pearson,spearman}_reconstruct_trans_by_<trans>_norm_by_<N>.feather, config.json, report.txt
    shiftformer.choose_norm(counts)          # only the rule (which normalization would be used)

Optional (not run by default): the 30-combination search
    result = shiftformer.choose_combination(train, test)   # raw matrix, dropout matrix
"""
from .pipeline import (ChooseCombinationResult, StopForUser, choose_combination,  # noqa: F401
                       load_result, merge_results)
from .preprocess import NORMALIZATIONS, TRANSFORMATIONS, all_combos  # noqa: F401
from .rule import choose_norm, input_features  # noqa: F401
from .workflow import RunResult, run  # noqa: F401

__version__ = "0.3.0"
