Figure 1 - MIDGaP pipeline
Script:  figures/fig1_pipeline.py   Palette: ASE file RULES/Ocean Sunset.ase
Size:    174 mm final, print scale 0.70, min text 7.0 pt; journal files 174.0 x 77.8 mm
QC:      overlaps 0 | legends outside | palette OK | exports verified (6 masters + 4 journal files)
Data:    none (schematic)
Encodes: schematic; solid = forward pass, dotted = training-time KL anchor, dashed = inference-time sampling
SVIS exceptions: diagram: no data axes, 2 pt outlines on boxes (SVIS 6, 11)

Figure 2 - Gate priors by criterion
Script:  figures/fig2_gate_priors.py   Palette: ASE file RULES/Ocean Sunset.ase
Size:    174 mm final, print scale 0.71, min text 8.5 pt; journal files 174.0 x 140.4 mm
QC:      overlaps 0 | legends outside | palette OK | exports verified (6 masters + 4 journal files)
Data:    cache/<dataset>_dcmi.json through code/dcmi.py gate_prior()
Encodes: prior weight per meta-path; deterministic, no error bars
SVIS exceptions: horizontal bars so that meta-path names stay horizontal (SVIS 7.3)

Figure 3 - Gate criteria versus the free gate
Script:  figures/fig3_gate_criteria.py   Palette: ASE file RULES/Ocean Sunset.ase
Size:    174 mm final, print scale 0.71, min text 8.5 pt; journal files 174.0 x 77.3 mm
QC:      overlaps 0 | legends outside | palette OK | exports verified (6 masters + 4 journal files)
Data:    results/ablation_gate_<dataset>.jsonl
Encodes: bars = mean over n = 3 seeds; error bars = 1 SD; circles = seeds; dashed line = free-gate mean
SVIS exceptions: none

Figure 4 - Accuracy by user-activity quintile
Script:  figures/fig4_activity_quintiles.py   Palette: ASE file RULES/Ocean Sunset.ase
Size:    174 mm final, print scale 0.71, min text 8.5 pt; journal files 174.1 x 70.4 mm
QC:      overlaps 0 | legends outside | palette OK | exports verified (6 masters + 4 journal files)
Data:    results/per_user_npz/<dataset>_<model>_s<seed>.npz, results/train_activity.json
Encodes: line = mean over n = 5 seeds; band = 1 SD; quintiles of training activity among evaluated users
SVIS exceptions: independent y scales across panels, all from zero (SVIS 7.4 exception, stated in caption)

Figure 5 - Predictive variance diagnostic
Script:  figures/fig5_uncertainty.py   Palette: ASE file RULES/Ocean Sunset.ase
Size:    174 mm final, print scale 0.71, min text 8.5 pt; journal files 174.0 x 73.1 mm
QC:      overlaps 0 | legends outside | palette OK | exports verified (6 masters + 4 journal files)
Data:    results/uncertainty_ml1m.json, results/per_user_npz/ml1m_LightGCN_s<seed>.npz
Encodes: (a) mean +/- 1 SD across users per decile; (b) mean +/- t-based 95% CI per decile
SVIS exceptions: single-series panels: no legend (SVIS 8, level 1)

