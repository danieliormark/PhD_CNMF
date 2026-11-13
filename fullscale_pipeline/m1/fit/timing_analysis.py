# Timing test analysis (ticket 107): seconds per epoch before the warm-up (epochs 50–300, terms off in every arm) and
# after it (epochs 350–900, terms on), per loss, K and arm, from the logged timestamps (every 50 epochs); the cost of
# each term = its arm's post-warm-up seconds per epoch ÷ the "off" arm's.
# Usage: python -m fit.timing_analysis --root /mnt/hum01-rds/Basov/p91688di/m1_fits
import argparse, glob, json, os
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); a = ap.parse_args()
    rows = {}
    for p in glob.glob(os.path.join(a.root, "timing_terms", "*", "result.json")):
        r = json.load(open(p)); j = json.load(open(os.path.join(os.path.dirname(p), "spec.json")))["job"]
        t = {l["epoch"]: l["seconds"] for l in r["logs"]}
        pre = (t[300] - t[50]) / 250 if 300 in t and 50 in t else None
        post = (t[850] - t[350]) / 500 if 850 in t and 350 in t else None
        rows[(j["beta"], j["K"], j["arm"]["timing"])] = (pre, post, r["epochs"])
    for (beta, K, arm), (pre, post, ep) in sorted(rows.items()):
        off = rows.get((beta, K, "off"), (None, None, None))[1]
        print(f"{'squared error' if beta == 2 else 'KL':13s} K={K:2d} {arm:14s} epochs {ep}  s/epoch before {pre:.3f}  after {post:.3f}"
              + (f"  × off {post / off:.2f}" if off and post else ""))
if __name__ == "__main__":
    main()
