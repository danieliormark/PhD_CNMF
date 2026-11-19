# Generic child hyperedges — step 2: their role in the existing fits (ticket 107; owner 2026-11-19: "I approve all 3
# points"). Classes from fit/generic_hyperedges.py (DRAFT rule, pending the owner's review): G genre, F focal, ? ambiguous,
# S substantive. Saved fits only, hold-out group 1, KL: the decided setting (start test 2, S3 10 %, K 13, seeds 1–10)
# and pass-1 fits (random social start) at K 8 / 13 / 20 / 32 for the change with K. Readings are numbering-free.
#   (a) composition: each community's child-side tie weight split by class (each visible child×article tie split among
#       communities on the child side by the fitted model — KL responsibilities — summed with the weighted values, and
#       with raw term frequencies); per fit: median and largest share of G over communities; the corpus-wide G share
#       is the reference (an even spread puts every community near it);
#   (b) mixing: tie-weighted mean entropy (bits) of the hyperedges' weighted memberships, by class; and how alike
#       communities look semantically: mean pairwise cosine between communities' child-side usage vectors (tie weight
#       per hyperedge), all hyperedges vs G removed vs G, F and ? removed;
#   (c) held-out existence on hidden child×article cells split by the hyperedge's class: within-class AP (as the
#       decided score) on the subset, with its chance level.
# Usage: python -m fit.generic_role --out fit/results/stage0/generic_role.json
import argparse, glob, json, math, os
import numpy as np
from . import data as D, measures as M
from .aggregate import load_fit, load_UZ, latest_fits
from .generic_hyperedges import labels

ROOT = "/mnt/hum01-rds/Basov/p91688di/m1_fits"
CLASSES = ("G", "F", "?", "S")


def bits(P):
    return -(P * np.log2(np.clip(P, 1e-300, None))).sum(1)


def cos_mean(Mx):
    n = np.linalg.norm(Mx, axis=1, keepdims=True); X = Mx / np.maximum(n, 1e-300); C = X @ X.T
    K = C.shape[0]; return float(C[~np.eye(K, dtype=bool)].mean())


def one(f, ca, lab, tf):
    U, Z = load_UZ(f["dir"]); rel = {k: tuple(v) for k, v in f["result"]["relations"].items()}
    A, B, z = U["article"], U["child"], Z["child_article"]; K = z.shape[0]
    r, c, v = ca.tie_row, ca.tie_col, ca.tie_val
    AZ = A[r] @ z; xh = (AZ * B[c]).sum(1); resp = AZ * B[c] / xh[:, None]           # ties × K (child-side community)
    out = {"K": K}
    cls = lab[c]
    for wname, w in (("weighted", v), ("raw", tf)):
        tot = (resp * w[:, None]).sum(0)
        g = (resp * (w * (cls == "G"))[:, None]).sum(0) / np.maximum(tot, 1e-300)
        out[f"G_share_median_{wname}"] = float(np.median(g)); out[f"G_share_max_{wname}"] = float(g.max())
        out[f"G_share_corpus_{wname}"] = float(w[cls == "G"].sum() / w.sum())
        for k in ("F", "?"):
            s = (resp * (w * (cls == k))[:, None]).sum(0) / np.maximum(tot, 1e-300)
            out[f"{k}_share_max_{wname}"] = float(s.max())
    P = M.weighted_memberships(U, Z, rel)["child"]["P"]; H = bits(P)
    for k in CLASSES:
        m = cls == k
        out[f"entropy_{k}"] = float((H[c[m]] * v[m]).sum() / v[m].sum()) if m.any() else None
    use = np.zeros((K, ca.shape[1])); np.add.at(use.T, c, resp * v[:, None])
    keepG = lab != "G"; keepS = lab == "S"
    out["cos_all"] = cos_mean(use); out["cos_without_G"] = cos_mean(use[:, keepG]); out["cos_S_only"] = cos_mean(use[:, keepS])
    ht = ((A[ca.hid_tie_row] @ z) * B[ca.hid_tie_col]).sum(1); he = ((A[ca.hid_empty_row] @ z) * B[ca.hid_empty_col]).sum(1)
    n_empty_obs = int(ca.obs_row_mask.sum()) * ca.shape[1] - len(ca.tie_row) - len(ca.hid_tie_row)
    scale = math.floor(0.10 * n_empty_obs) / len(he)
    ap, _, ch = M.within_class_ap(ht, he, ca, scale)
    out["existence_all"], out["chance_all"] = ap, ch
    for k in CLASSES:
        kt, ke = lab[ca.hid_tie_col] == k, lab[ca.hid_empty_col] == k
        ap, left, ch = M.within_class_ap(ht, he, ca, scale, keep_t=kt, keep_e=ke)
        out[f"existence_{k}"], out[f"chance_{k}"], out[f"hidden_ties_{k}"] = ap, ch, int(kt.sum())
    return out


def run():
    build = D.Build(D.BUILD); lab = labels(build); rep = {}
    raw = build.load("raw_child_article_tf")
    allf = {s: [load_fit(os.path.dirname(p)) for p in glob.glob(os.path.join(ROOT, s, "*", "measures.json"))]
            for s in ("start_test2", "stage0_pass1")}
    for a in (1, 2):
        ca = next(rd for rd in D.fit_data(build, a, group=0) if rd.name == "child_article")
        tf = np.asarray(raw[ca.tie_row, ca.tie_col]).ravel().astype(float)
        sets = {"decided_S3_10_K13": [f for f in latest_fits(allf["start_test2"]) if f["job"]["approach"] == a and f["job"]["group"] == 0
                                      and f["job"].get("offdiag_level") == 0.1 and f["job"].get("offdiag_semantic")]}
        for K in (8, 13, 20, 32):
            sets[f"pass1_K{K}"] = [f for f in latest_fits(allf["stage0_pass1"]) if f["job"]["approach"] == a and f["job"]["group"] == 0
                                   and f["job"]["beta"] == 1 and f["job"]["K"] == K and not f["job"].get("arm") and not (f["job"].get("terms") or {})]
        rep[f"approach_{a}"] = {}
        print(f"approach {a} — medians over fits")
        for name, fs in sets.items():
            rows = [one(f, ca, lab, tf) for f in fs]
            med = {k: float(np.median([x[k] for x in rows if x[k] is not None])) for k in rows[0] if any(x[k] is not None for x in rows)}
            rep[f"approach_{a}"][name] = {"fits": len(rows), "median": med}
            print(f"  {name} ({len(rows)} fits): G share of a community's child-side tie weight — median {med['G_share_median_weighted']:.3f}, "
                  f"largest {med['G_share_max_weighted']:.3f} (corpus {med['G_share_corpus_weighted']:.3f}); raw: median {med['G_share_median_raw']:.3f}, "
                  f"largest {med['G_share_max_raw']:.3f} (corpus {med['G_share_corpus_raw']:.3f}); largest F {med['F_share_max_weighted']:.3f}, ? {med['?_share_max_weighted']:.3f}")
            print(f"     membership entropy (bits, tie-weighted): " + "  ".join(f"{k} {med['entropy_'+k]:.3f}" for k in CLASSES)
                  + f" | mean cosine between communities: all {med['cos_all']:.4f}, without G {med['cos_without_G']:.4f}, S only {med['cos_S_only']:.4f}")
            print(f"     held-out existence (within-class AP / chance): all {med['existence_all']:.4f}/{med['chance_all']:.4f}  "
                  + "  ".join(f"{k} {med['existence_'+k]:.4f}/{med['chance_'+k]:.4f} (n {int(med['hidden_ties_'+k])})" for k in CLASSES), flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    json.dump(run(), open(ap.parse_args().out, "w"), indent=1)


if __name__ == "__main__":
    main()
