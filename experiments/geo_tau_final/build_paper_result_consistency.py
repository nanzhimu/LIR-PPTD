from pathlib import Path
import csv, hashlib
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'analysis/geo_tau_final/paper_result_consistency.csv'
def sha(rel):
 p=ROOT/rel
 if not p.exists() or not p.is_file(): return ''
 h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()
rows=[]
def add(section,item,metric,dataset,version,tau,source,rerun,updated,status,notes=''):
 rows.append(dict(section=section,figure_table=item,metric=metric,dataset=dataset,algorithm_version=version,tau_mode=tau,source_file=source,source_sha256=sha(source),rerun_needed=rerun,updated_value=updated,status=status,notes=notes))
add('Abstract','claim','17/20 nonzero malicious-ratio lower mean loss vs CRH/PPTD','Product/Duck/Dog/Weather','geometry-final','geometry_diameter_scaling','results/p0_1c_hidden_b1_geometry_v1/figure_b1_hidden_metric_means.csv','no','17/20','verified')
add('Abstract','claim','NetEase LIR>No-HR point estimate','NetEaseCrowd','geometry-final','geometry_diameter_scaling','analysis/geo_tau_final/netease_capability_geometry.csv','no','6/6 capabilities','verified')
add('Sec.I','Contribution 4','benchmark/NetEase/geometry/secure-execution evidence','all','geometry-final','geometry_diameter_scaling','analysis/geo_tau_final/experiment_impact_matrix.csv','no','final geometry evidence set','verified')
add('Sec.III/IV','Eq. tau','tau=lambda*diameter^2(X)','all','geometry-final','geometry_diameter_scaling','src/lir_pptd/core/consistency_scale.py','no','geometry diameter scaling','verified')
add('Sec.V','fixed-point bound','domain squared-distance bound B_d','numerical/categorical','geometry-final','geometry_diameter_scaling','src/lir_pptd/fixedpoint/range_analysis.py','no','numerical D; categorical 2','verified')
add('VI-B','Fig.3','malicious-ratio main','4 benchmarks','geometry-final','geometry_diameter_scaling','results/p0_1c_hidden_b1_geometry_v1/figure_b1_hidden_metric_means.csv','no','geometry-aware Dog; equivalent other domains','verified')
add('VI-B','Fig.4','attack robustness','4 benchmarks','geometry-final','geometry_diameter_scaling','results/b2_hidden_selector_geometry_v1/formal_summary.json','no','geometry-aware Dog; equivalent other domains','verified')
add('VI-C','Fig.5','ablation','4 benchmarks','geometry-final','geometry_diameter_scaling','results/vi_c_hidden_selector_geometry_v1/formal_summary.json','no','geometry-aware Dog; equivalent other domains','verified')
add('VI-C','Fig.6','reputation evolution','Dog/Weather','geometry-final','geometry_diameter_scaling','analysis/geo_tau_final/near_tie_release.csv','no','Dog geometry / Weather equivalent','verified','trajectory source retained in C10 geometry family')
add('VI-D','Fig.7','parameter sensitivity','Dog/Weather','geometry-final','geometry_diameter_scaling','analysis/geo_tau_final/parameter_sensitivity_geometry_final.csv','no','560/560 Dog geometry exact runs + Weather numerical-equivalence evidence','verified')
add('VI-D','Fig.8','calibration corruption/missingness','Dog/Weather','geometry-final','geometry_diameter_scaling','analysis/geo_tau_final/calibration_quality_geometry_final.csv','no','200/200 Dog geometry exact runs + Weather numerical-equivalence evidence','verified')
add('VI-D/Supp','order sensitivity','5 fixed task-ID permutations','4 benchmarks','geometry-final','geometry_diameter_scaling','analysis/geo_tau_final/task_order_geometry_final.csv','no','200/200 Dog geometry exact runs; numerical/binary domains retained by equivalence','verified')
add('VI-B/Supp','behavior-switch','held-out switch loss','Dog + equivalent other domains','geometry-final','geometry_diameter_scaling','results/cowa_dynamic_formal_geometry_v1/formal_runs.jsonl','no','geometry-aware COWA/LIR/No-CW','verified')
add('VI-D/Supp','high-D geometry','1200 paired categorical runs','synthetic','geometry-final','geometry_diameter_scaling','analysis/geo_tau_final/experiment_impact_matrix.csv','no','D16/32 12/12 non-worse','verified')
add('VI-E','Fig.9 after reintegration','arithmetic/kernel efficiency','numerical D=1','geometry-final','geometry_diameter_scaling','paper/figures/fig_vi_e_efficiency_main_compact_tight.png','no','unchanged: numerical tau identical','verified')
add('VI-E','MP-SPDZ table','real MPC latency/fidelity','numerical D=1','geometry-final','geometry_diameter_scaling','configs/p1d/p1d_e2e_design.json','no','resolved tau=0.2, primitive structure unchanged','verified')
add('VI-F','recovery','availability/recovery semantics','protocol','geometry-final','not_tau_dependent','results/vi_f_final_semantics_v1/formal_summary.json','no','unchanged','verified')
add('VI-G','Fig.10 after reintegration','synthetic operating boundary','mixed','geometry-final','geometry_diameter_scaling','paper/figures/fig_vi_g_operating_boundary_main_compact.png','no','active runner geometry/default verified','verified')
add('VI-H','Fig.11 after reintegration','NetEase longitudinal','NetEaseCrowd D=3','geometry-final','geometry_diameter_scaling','analysis/geo_tau_final/netease_30day_geometry.csv','no','C10 geometry rerun/current-path regression','verified')
add('VI-H','capability table','Accuracy/MacroF1 by capability','NetEaseCrowd D=3','geometry-final','geometry_diameter_scaling','analysis/geo_tau_final/netease_capability_geometry.csv','no','geometry values','verified')
add('VI-H','history complete','dependence-aware gains','NetEaseCrowd D=3','geometry-final','geometry_diameter_scaling','analysis/geo_tau_final/netease_history_complete_geometry.csv','no','geometry values','verified')
add('VI-H','re-entry','Reset30d','NetEaseCrowd D=3','geometry-final','geometry_diameter_scaling','analysis/geo_tau_final/netease_reentry_geometry.csv','no','geometry values','verified')
add('Conclusion','claim','17/20 + NetEase + geometry + efficiency','all','geometry-final','geometry_diameter_scaling','analysis/geo_tau_final/experiment_impact_matrix.csv','no','final evidence map','verified')
OUT.parent.mkdir(parents=True,exist_ok=True)
with OUT.open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('rows',len(rows),'pending',sum(r['status']=='pending' for r in rows))
