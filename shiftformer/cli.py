"""Command line (thin wrapper around the Python API):
    shiftformer run    --input RAW --out_dir OUT [--reference REF]          (default workflow)
    shiftformer choose --train RAW --test DROPOUT --out_dir OUT ...         (optional: 30-combination search)
    shiftformer merge  --parts DIR1 DIR2 ... --out_dir OUT                  (optional: merge choose parts)
"""
import argparse
import json
import os
import sys

from . import pipeline as PL
from . import workflow as RUN


def main(argv=None):
    ap = argparse.ArgumentParser(prog="shiftformer",
                                 description="Transformer denoise with log2(count+2) and a rule-chosen normalization")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="default: denoise one raw count matrix; writes recon_pearson / recon_spearman")
    r.add_argument("--input", required=True, help="raw count matrix file (cells x genes, first column = cell IDs)")
    r.add_argument("--out_dir", required=True,
                   help="writes recon_pearson.feather, recon_spearman.feather, config.json, report.txt")
    r.add_argument("--reference", default=None, help="optional matrix of the same cells, used only for evaluation")
    r.add_argument("--orientation", default="cells_x_genes", choices=["cells_x_genes", "genes_x_cells"])
    r.add_argument("--pearson_norm", default=None, help="override the rule (1000/10000/100000/1000000/no_norm)")
    r.add_argument("--spearman_norm", default=None, help="override the rule (default rule: 1000000)")
    r.add_argument("--no_clip", action="store_true", help="keep negative values after the inverse transformation")
    r.add_argument("--save_model_space", action="store_true",
                   help="also write model_output_<criterion>.feather (log2 space, as the original Step3)")
    r.add_argument("--save_dtype", default="float32", choices=["float32", "float64"])
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    r.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) or None)

    s = sub.add_parser("choose", help="optional (not default): 30 transformation x normalization combinations "
                                      "on a train (raw) and test (dropout) matrix")
    s.add_argument("--train", required=True, help="raw count matrix file (no dropout/norm/trans)")
    s.add_argument("--test", required=True, help="dropout count matrix file (same cells/genes)")
    s.add_argument("--out_dir", required=True, help="writes summary.tsv, report.txt, run_log.json")
    s.add_argument("--orientation", default="cells_x_genes", choices=["cells_x_genes", "genes_x_cells"])
    s.add_argument("--combinations", default="all", help="'all' or comma list of indices 0..29")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--save_dtype", default="float32", choices=["float32", "float64"])
    s.add_argument("--pipeline_genes", default=None,
                   help="optional reference gene set (.txt one ID per line, or a matrix file)")
    s.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    s.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) or None)
    m = sub.add_parser("merge", help="optional: merge results of different combination subsets of 'choose'")
    m.add_argument("--parts", nargs="+", required=True)
    m.add_argument("--out_dir", required=True)

    a = ap.parse_args(argv)
    if a.cmd == "run":
        try:
            RUN.run(a.input, reference=a.reference, out_dir=a.out_dir, orientation=a.orientation, seed=a.seed,
                    pearson_norm=a.pearson_norm, spearman_norm=a.spearman_norm, clip_negative=not a.no_clip,
                    save_dtype=a.save_dtype, save_model_space=a.save_model_space, device=a.device,
                    threads=a.threads)
        except PL.StopForUser as e:
            _stop(a.out_dir, e)
        sys.exit(0)
    if a.cmd == "choose":
        combos = None if a.combinations == "all" else [int(x) for x in a.combinations.split(",")]
        try:
            res = PL.choose_combination(a.train, a.test, a.orientation, combos, a.seed,
                                        save_dtype=a.save_dtype, pipeline_genes=a.pipeline_genes,
                                        device=a.device, threads=a.threads)
        except PL.StopForUser as e:
            _stop(a.out_dir, e)
    else:
        res = PL.merge_results([PL.load_result(d) for d in a.parts])
    res.save(a.out_dir)
    print(res.report)
    sys.exit(0 if res.final or a.cmd == "choose" and a.combinations != "all" else 1)


def _stop(out_dir, e):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "run_log.json"), "w") as f:
        json.dump(e.log, f, indent=2, default=str)
    print(f"[STOP] {e} (details: {os.path.join(out_dir, 'run_log.json')})", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
