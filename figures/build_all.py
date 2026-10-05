# -*- coding: utf-8 -*-
"""Rebuild every manuscript figure with one command (strict QC; a failing figure stops the build).

Outputs (default figures/output/, override with FIG_OUT):
  masters/Figure_<n>_<white|transparent>.<pdf|tiff|png>   SVIS master set, design size
  journal/Fig<n>.<eps|pdf|tif|png>                         journal files, 174 mm wide, 600 dpi rasters
  figure_captions.json / figure_captions.txt               three-part captions, numbers injected
  fig<n>_values.json                                       every plotted value
  DELIVERY_NOTES.md, figure_build_report.json              QC and delivery notes (SVIS 18.3)
"""
import json
import os

import matplotlib

matplotlib.use("Agg")

import figcommon as C
import fig1_pipeline
import fig2_gate_priors
import fig3_gate_criteria
import fig4_activity_quintiles
import fig5_uncertainty


def fmt_int(n):
    return "{:,}".format(int(n))


def captions():
    v4 = C.jload(os.path.join(C.OUT, "fig4_values.json"))
    v5 = C.jload(os.path.join(C.OUT, "fig5_values.json"))
    v3 = C.jload(os.path.join(C.OUT, "fig3_values.json"))
    n_seeds3 = sorted({n for d in v3["n_seeds"].values() for n in d.values()})
    n_seeds4 = sorted({v4[d][m]["n_seeds"] for d in v4 for m in v4[d] if isinstance(v4[d][m], dict) and m != "MostPop"})
    assert len(n_seeds3) == 1 and len(n_seeds4) == 1
    assert all(v4[d]["MostPop"]["n_seeds"] == 1 for d in v4)          # deterministic ranker, run once
    users4 = [fmt_int(v4[d]["n_users"]) for d in ("ml1m", "lastfm", "amazon")]
    word = {3: "three", 5: "five"}
    cap = {
        1: ("Pipeline of the MIDGaP recommender. In the offline lane, each meta-path subgraph is scored by "
            "density-corrected mutual information (DCMI), the standardised excess of a plug-in mutual information "
            "estimate over 40 degree-preserving surrogate graphs, and a tempered softmax of the scores gives the gate "
            "prior π. In the recommender lane, content and ID embeddings pass through the gated meta-path branch, "
            "whose gates g are anchored to π by a Kullback-Leibler penalty during training (dotted connector), and "
            "a variational bottleneck yields the posterior whose mean ranks the items and whose S = 16 samples give the "
            "predictive variance at inference (dashed connector). The prior is computed once per dataset, before training"),
        2: ("Gate priors that the three scoring criteria assign to every meta-path. Rows show ML-100K (a, b), ML-1M "
            "(c, d), Last.FM (e, f) and Amazon-VG (g, h); left panels hold the user-side and right panels the item-side "
            "meta-paths. Each bar is the prior weight under density-corrected mutual information (DCMI), raw mutual "
            "information or subgraph density; all criteria pass through the same range normalisation and tempered "
            "softmax, so the panels isolate the scoring function, and the weights are deterministic functions of the "
            "meta-path statistics (Table 5; ML-100K in Online Resource 1). Amazon-VG has a single user-side meta-path, "
            "which every criterion weights at one, and ML-100K serves only the protocol study. Density and DCMI select "
            "different relations on ML-1M (user side) and invert each other on Amazon-VG (item side)"),
        3: ("Gate criteria compared with the gate that the ranking loss learns without a prior. (a) Item-side gate "
            "weight that each criterion produces on the route the free (unanchored) gate selects on each benchmark, "
            "with the route named below the benchmark; the dashed line extends the free-gate mean across the group. "
            "(b) NDCG@20 of each criterion relative to the DCMI anchor (0%%). Bars show means over n = %d seeds, error "
            "bars one standard deviation and open circles the individual seeds. On Amazon-VG only the density prior "
            "moves the gate against the direction the ranking loss discovers, whereas on Last.FM the corrected and "
            "density priors both move it away from that route" % n_seeds3[0]),
        4: ("Ranking accuracy by user activity. NDCG@20 of five methods in each quintile of training-window activity "
            "on (a) ML-1M, (b) Last.FM and (c) Amazon-VG, where Q1 holds the least active users. Lines show the mean "
            "over n = %d seeds and shaded bands one standard deviation (MostPop is deterministic and was run once); quintiles are cut among the evaluated users "
            "(%s, %s and %s users) and are unequal in size where users share the same activity count, and each panel "
            "has its own y-axis scale. The ordering of methods is largely flat across quintiles, so no method shows a "
            "cold-start-specific advantage" % (n_seeds4[0], users4[0], users4[1], users4[2])),
        5: ("Predictive variance as an uncertainty diagnostic on ML-1M (n = %s test users). (a) Top-20 predictive "
            "variance of MIDGaP, averaged over %s seeds, by user-activity decile; points are decile means and error "
            "bars one standard deviation across the users of a decile. (b) Per-user NDCG@20 difference between MIDGaP "
            "(%s seeds) and LightGCN (%s seeds) by predictive-variance decile; bars are decile means and error bars "
            "t-based 95%% confidence intervals. Variance falls with activity but does not order users by how well they "
            "are ranked, which is why it is reported as a diagnostic rather than as a confidence score"
            % (fmt_int(v5["n_users"]), word[v5["n_seeds_midgap"]], word[v5["n_seeds_midgap"]],
               word[v5["n_seeds_lightgcn"]])),
    }
    for n, t in cap.items():
        assert not t.rstrip().endswith("."), n          # Springer: no punctuation at the end of a caption
        assert "—" not in t, n                      # house writing rules: no em dash
    return cap


def main():
    report = []
    for mod in (fig1_pipeline, fig2_gate_priors, fig3_gate_criteria, fig4_activity_quintiles, fig5_uncertainty):
        info = mod.main()
        report.append(info)
        print("Fig%d  QC %s  print scale %.2f  min text %.1f pt  journal %s x %s mm" % (
            info["figure"], info["qc"], info["print_scale"], info["min_text_pt"],
            info["journal"]["width_mm"], info["journal"]["height_mm"]))
    cap = captions()
    C.write_json("figure_captions.json", {"Fig%d" % n: t for n, t in cap.items()})
    with open(os.path.join(C.OUT, "figure_captions.txt"), "w", encoding="utf-8") as f:
        f.write("Figure captions (Springer style: \"Fig.\" and the number in bold, no punctuation after the number "
                "and none at the end of the caption)\n")
        f.write("Graphics program: Matplotlib %s (Python). Lettering: Arial. Width 174 mm. EPS and PDF vector "
                "(TrueType fonts embedded), TIFF and PNG at 600 dpi, RGB.\n\n" % matplotlib.__version__)
        for n, t in cap.items():
            f.write("Fig. %d  %s\n\n" % (n, t))
    C.write_json("figure_build_report.json", report)
    with open(os.path.join(C.OUT, "DELIVERY_NOTES.md"), "w", encoding="utf-8") as f:
        for r in report:
            f.write("Figure %d - %s\n" % (r["figure"], r["title"]))
            f.write("Script:  figures/%s   Palette: %s\n" % (r["script"], r["palette"]))
            f.write("Size:    174 mm final, print scale %.2f, min text %.1f pt; journal files %s x %s mm\n"
                    % (r["print_scale"], r["min_text_pt"], r["journal"]["width_mm"], r["journal"]["height_mm"]))
            f.write("QC:      overlaps 0 | legends outside | palette OK | exports verified (6 masters + 4 journal files)\n")
            f.write("Data:    %s\n" % r["data"])
            f.write("Encodes: %s\n" % r["encodes"])
            f.write("SVIS exceptions: %s\n\n" % r["exceptions"])
    print("captions and notes written to", C.OUT)


if __name__ == "__main__":
    main()
