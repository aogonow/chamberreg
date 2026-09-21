"""Registers the CAD model of an ionization chamber to its CT scan and scores the result."""
import argparse
import json
import sys

from .pipeline import format_line, register_chamber, run_dataset, summarize, write_summary

def main(argv=None):
    ap = argparse.ArgumentParser(prog="chamberreg", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="register every chamber of a dataset")
    run.add_argument("dataset", help="directory containing sinograms/<chamber>_recon_sino.hdf5")
    run.add_argument("--out", default="results")
    run.add_argument("--chambers", nargs="*", help="subset of chambers (default: all)")
    run.add_argument("--jobs", type=int, default=1, help="chambers processed in parallel")
    run.add_argument("--threads", type=int, default=8, help="threads per chamber for the score scans")

    one = sub.add_parser("one", help="register a single chamber and print the result as JSON")
    one.add_argument("dataset")
    one.add_argument("chamber")
    one.add_argument("--threads", type=int, default=8)

    summ = sub.add_parser("summarize", help="rebuild summary.json / summary.md from per-chamber results")
    summ.add_argument("results")

    a = ap.parse_args(argv)
    if a.command == "run":
        s = run_dataset(a.dataset, a.out, a.chambers, a.jobs, a.threads)
        print(open(f"{a.out}/summary.md").read())
        return 0 if s["n_failed"] == 0 else 1
    if a.command == "one":
        r = register_chamber(a.dataset, a.chamber, a.threads)
        print(format_line(r), file=sys.stderr)
        print(json.dumps(r, indent=1))
        return 0
    if a.command == "summarize":
        write_summary(summarize(a.results), a.results)
        print(open(f"{a.results}/summary.md").read())
        return 0

if __name__ == "__main__":
    sys.exit(main())
