"""A3 part 2, step 3 (RL-203): P6 v2 (pmc_preprocessing/coref_resolution_v2.py, unchanged) on the articles with an
addition (a3p2_prepare.py), with the v3.1 matcher (v3 + G3's hub spellings). A copy of inc_p6.py (RL-165) with the inputs
of LCS/a3p2/ and the matcher swapped; run locally on incline (a few hundred articles). The script's module-level paths are pointed at the unit:
    gap      input LCS/a3p2/p6_in_gap/focal_extractions_v2.jsonl, context texts and regions from the staging tree
    current  input LCS/a3p2/p6_in_current/focal_extractions_v2.jsonl, the current corpus's own texts and regions
    python a3p2_p6.py --unit gap|current <coref_resolution_v2.py arguments: --shard K --nshards N --outdir DIR --resume ...>
"""
import sys, importlib.util
R = "/mnt/hum01-rds/Basov/p91688di/"
PP, LCS = R + "pmc_preprocessing/", R + "llm_corpus_staging/"
UNITS = {"gap": dict(LCS=LCS + "a3p2/p6_in_gap/", PP=LCS + "gap_stage/pmc_preprocessing/",
                     CLEAN=LCS + "gap_stage/llm_corpus_staging/clean_corpus_v2/"),
         "current": dict(LCS=LCS + "a3p2/p6_in_current/", PP=PP, CLEAN=LCS + "clean_corpus_v2/")}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m


if __name__ == "__main__":
    i = sys.argv.index("--unit"); unit = sys.argv[i + 1]; rest = sys.argv[1:i] + sys.argv[i + 2:]
    sys.path.insert(0, PP)
    load("focal_terms", PP + "focal_terms.py")                         # v3 imports v2 under this name
    load("focal_terms_v3", PP + "focal_terms_v3.py")                  # v3.1 imports v3 under this name
    sys.modules["focal_terms"] = load("focal_terms_v3_1", PP + "focal_terms_v3_1.py")
    m = load("coref_resolution_v2", PP + "coref_resolution_v2.py")
    for k, v in UNITS[unit].items():
        setattr(m, k, v)
    sys.argv = ["coref_resolution_v2.py"] + rest
    m.main()
