export SDIR=out/sampler
export MODIR=out/models.synth
export SODIR=out/samples.synth
export Np=50
export Ntr=1
export OPID=3,8,12
export APID=4,13
export seed=1

# SAMPLER
python experiment/real_world/treatment/run_gprpp_joint.py --model_dir ${SDIR} --patient_ids ${APID} --action_components ao --maxiter 500 --n_day_train 3
python evaluate/plot_figure9.py --model_dir ${SDIR} --patient_ids ${APID} --action_components ao
python experiment/real_world/outcome/run_outcome_multiple_marked_hierarchical.py --outcome_model_dir ${SDIR} --patient_ids ${OPID} --n_day_train 2 --n_day_test 1
python evaluate/plot_figure10.py --outcome_model_dir ${SDIR} --patient_ids ${OPID} --n_day_train 2 --n_day_test 1
# SYNTH EXPERIMENT
python experiment/synth/sample_train_data.py --sampler_dir ${SDIR} --model_output_dir ${MODIR} --sample_output_dir ${SODIR} --n_patient $Np --n_day_train $Ntr --n_day_test 1 --seed $seed
Rscript models/benchmarks/hua/doct/run_script.R ${SODIR} 1  # Run hua benchmark code in R
python experiment/synth/run_synth_exp.py --sampler_dir ${SDIR} --model_output_dir ${MODIR} --sample_output_dir ${SODIR} --n_patient $Np --n_day_train $Ntr --n_day_test 1 --seed $seed
python evaluate/plot_figure11.py --sample_dir ${SODIR}/joint.seed1/np$Np.ntr$Ntr.pa${APID}.po${OPID}.no40.tTrue
python evaluate/plot_figure12.py --sample_dir ${MODIR}/joint.seed$seed/np$Np.ntr$Ntr.pa${APID}.po${OPID}.no40.tTrue/evaluate/policy_interventional
python evaluate/report_mse_multiple.py --sampler_dir ${SDIR} --model_output_dir ${MODIR} --sample_output_dir ${SODIR} --n_patient $Np --n_day_train $Ntr --n_day_test 1 --seeds 1,2,3,4,5,6,7,8,9,10
