# Stage-1 journal arms J4 / J5 (data.journal_values; 2026-11-26): masses, masks, per-journal totals, other relations unchanged.
import numpy as np
from fit import data as D


def _rds(**kw):
    return {rd.name: rd for rd in D.fit_data(D.Build(D.BUILD), 1, group=0, **kw)}


def test_journal_forms():
    base = _rds()
    J = base["article_journal"]
    assert J.journal is None
    for form in D.JOURNAL_FORMS:
        for mass in ("restore", "frobenius"):
            r = _rds(journal_form=form, journal_mass=mass)
            j = r["article_journal"]
            for k in base:
                if k != "article_journal":
                    assert np.array_equal(r[k].tie_val, base[k].tie_val) and r[k].weight == base[k].weight
            if mass == "restore":
                assert abs(j.tie_val.sum() - J.tie_val.sum()) < 1e-9          # J2's visible journal mass kept
            else:
                assert abs(np.sqrt((j.tie_val ** 2).sum()) - 1.0) < 1e-9       # Frobenius norm 1
            n = np.bincount(j.tie_col, minlength=j.shape[1])
            if form != "idf":
                assert n[n > 0].min() >= 2                                     # singleton-journal ties missing
                assert not j.obs_row_mask[np.setdiff1d(J.tie_row, j.tie_row)].any()
            if form == "inv_n":
                tot = np.bincount(j.tie_col, weights=j.tie_val, minlength=j.shape[1])[n > 0]
                assert np.allclose(tot, tot[0])                                # equal total per journal
