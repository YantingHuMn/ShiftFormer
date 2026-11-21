"""Command line (thin wrapper around the Python API):
    shiftformer choose --train RAW --test DROPOUT --out_dir OUT [--combinations 0,1,...]
    shiftformer merge  --parts DIR1 DIR2 ... --out_dir OUT
"""
import argparse
import json
import os
import sys

from . import pipeline as PL


def main(argv=None):
    ap = argparse.ArgumentParser(prog="shiftformer",
                                 description="Transformation x normalization selection with Transformer denoise (v1)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("choose", help="run the selection on a train (raw) and test (dropout) matrix")
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
    m = sub.add_parser("merge", help="merge results of different combination subsets")
    m.add_argument("--parts", nargs="+", required=True)
    m.add_argument("--out_dir", required=True)

    a = ap.parse_args(argv)
    if a.cmd == "choose":
        combos = None if a.combinations == "all" else [int(x) for x in a.combinations.split(",")]
        try:
            res = PL.choose_combination(a.train, a.test, a.orientation, combos, a.seed,
                                        save_dtype=a.save_dtype, pipeline_genes=a.pipeline_genes,
                                        device=a.device, threads=a.threads)
        except PL.StopForUser as e:
            os.makedirs(a.out_dir, exist_ok=True)
            with open(os.path.join(a.out_dir, "run_log.json"), "w") as f:
                json.dump(e.log, f, indent=2, default=str)
            print(f"[STOP] {e} (details: {os.path.join(a.out_dir, 'run_log.json')})", file=sys.stderr)
            sys.exit(2)
    else:
        res = PL.merge_results([PL.load_result(d) for d in a.parts])
    res.save(a.out_dir)
    print(res.report)
    sys.exit(0 if res.final or a.cmd == "choose" and a.combinations != "all" else 1)


if __name__ == "__main__":
    main()
