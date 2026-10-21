"""Incremental run, step 3 (RL-165): P6 v2 (pmc_preprocessing/coref_resolution_v2.py, unchanged) on one unit of the
incremental run, with the v3 matcher. The script's module-level paths are pointed at the unit:
    new      input LCS/inc_run/p6_new_in/focal_extractions_v2.jsonl (the 15,563 new articles, cleaned by inc_prepare.py),
             context texts and regions from the staging tree (LCS/gap_stage/)
    current  input LCS/inc_run/p6_cur_in/focal_extractions_v2.jsonl (current articles with additions), the current
             corpus's own context texts and regions
    python inc_p6.py --unit new|current <coref_resolution_v2.py arguments: --shard K --nshards N --outdir DIR --resume ...>
"""
import sys, importlib.util
R = "/mnt/hum01-rds/Basov/p91688di/"
PP, LCS = R + "pmc_preprocessing/", R + "llm_corpus_staging/"
UNITS = {"new": dict(LCS=LCS + "inc_run/p6_new_in/", PP=LCS + "gap_stage/pmc_preprocessing/",
                     CLEAN=LCS + "gap_stage/llm_corpus_staging/clean_corpus_v2/"),
         "current": dict(LCS=LCS + "inc_run/p6_cur_in/", PP=PP, CLEAN=LCS + "clean_corpus_v2/")}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m


if __name__ == "__main__":
    i = sys.argv.index("--unit"); unit = sys.argv[i + 1]; rest = sys.argv[1:i] + sys.argv[i + 2:]
    sys.path.insert(0, PP)
    load("focal_terms", PP + "focal_terms.py")                         # v3 imports v2 under this name
    sys.modules["focal_terms"] = load("focal_terms_v3", PP + "focal_terms_v3.py")
    m = load("coref_resolution_v2", PP + "coref_resolution_v2.py")
    for k, v in UNITS[unit].items():
        setattr(m, k, v)
    sys.argv = ["coref_resolution_v2.py"] + rest
    m.main()
