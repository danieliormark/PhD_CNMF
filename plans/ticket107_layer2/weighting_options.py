# Ticket 107, L2.1/L2.2 (2026-11-02): pull = relation weight × best share capturable at K = 4 (capturable_shares.txt),
# as % of the sum, for the domain-split options A-E, both approaches, journal variants (none, binary 0.2298, idf 0.1024,
# 1/sqrt(n) with singleton journals missing 0.0237). Pure arithmetic on the K = 4 shares.
B = {"cp": 0.0123, "ca": 0.0310, "auart": 0.0098, "auaff": 0.0281, "artaff": 0.0301, "art2": 0.0262}
J = {"none": None, "binary": 0.2298, "idf": 0.1024, "1/sqrt(n), singletons missing": 0.0237}
APPR = {1: ["auart", "auaff"], 2: ["artaff", "art2"]}
def weights(opt, soc):
    s = len(soc)
    if opt == "E equal per matrix": n = 2 + s; return {**{"cp": 1 / n, "ca": 1 / n}, **{m: 1 / n for m in soc}}
    if opt == "A 50/50, c×a semantic": return {"cp": .25, "ca": .25, **{m: .5 / s for m in soc}}
    if opt == "B 50/50, c×a social": return {"cp": .5, "ca": .5 / (s + 1), **{m: .5 / (s + 1) for m in soc}}
    if opt == "C 50/50, c×a half each":
        return {"cp": .5 / 1.5, "ca": .5 / 1.5 * .5 + .5 / (s + .5) * .5, **{m: .5 / (s + .5) for m in soc}}
    if opt == "D thirds": return {"cp": 1 / 3, "ca": 1 / 3, **{m: 1 / 3 / s for m in soc}}
for a, base in APPR.items():
    for jn, jb in J.items():
        print(f"--- approach {a}, journal {jn}")
        for opt in ["A 50/50, c×a semantic", "B 50/50, c×a social", "C 50/50, c×a half each", "D thirds", "E equal per matrix"]:
            soc = base + ([] if jb is None else ["j"]); b = dict(B, j=jb); w = weights(opt, soc)
            tot = sum(w[m] * b[m] for m in w); sh = {m: w[m] * b[m] / tot for m in w}
            print(f"  {opt:26s} " + " ".join(f"{m}:{w[m]:.3f}" for m in w) + f" | pull child×parent {sh['cp']:.0%}, "
                  f"child×article {sh['ca']:.0%}, social {sum(sh[m] for m in soc):.0%}" + (f" (journal {sh['j']:.0%})" if "j" in sh else ""))
