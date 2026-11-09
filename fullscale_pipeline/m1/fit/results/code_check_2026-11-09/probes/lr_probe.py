import sys, json, numpy as np
sys.path.insert(0, "/mnt/hum01-home01/p91688di/PhD_CNMF/fullscale_pipeline/m1")
from fit import data as D, worlds as Wd
from fit.solver import fit
from fit.run_code_check import summarise
version, beta, mode, lr, start = sys.argv[1], int(sys.argv[2]), sys.argv[3], float(sys.argv[4]), sys.argv[5]
rds, Ut, Zt = Wd.code_check_world(version, seed=1, build=D.Build())
kw = dict(init_U=Ut, init_Z=Zt) if start == "truth" else dict(start=start)
r = fit(rds, 6, beta, seed=1, max_epochs=2000, lr=lr, lr_mode=mode, **kw)
s = summarise(r, rds, Ut, Zt)
print(json.dumps({"args": sys.argv[1:], "conv": s["converged"], "epochs": s["epochs"], "loss": round(s["final_data_loss"], 6),
                  "maxrelerr": round(s["max_relative_error"], 4), "mincos": round(s["min_cosine"], 3),
                  "CA_within": s["child_article_within_share"], "CA_presence_min": min(s["child_article_presence"]["fit"])}))
