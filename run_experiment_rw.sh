export PIDS=0,1,2,3,4,5,6,7,8,9,10,11,12,13
export TMDIR=out/models.treatment
export OMDIR=out/models.outcome

python experiment/real_world/treatment/run_gprpp_joint.py --model_dir ${TMDIR}/fbao --patient_ids ${PIDS} --action_components bao --n_day_train 2 --n_day_test 1
python experiment/real_world/treatment/run_gprpp_joint.py --model_dir ${TMDIR}/fba --patient_ids ${PIDS} --action_components ba --n_day_train 2 --n_day_test 1
python experiment/real_world/treatment/run_gprpp_joint.py --model_dir ${TMDIR}/fbo --patient_ids ${PIDS} --action_components bo --n_day_train 2 --n_day_test 1
python experiment/real_world/treatment/run_gprpp_joint.py --model_dir ${TMDIR}/fao --patient_ids ${PIDS} --action_components ao --n_day_train 2 --n_day_test 1
python experiment/real_world/treatment/run_vbpp.py --model_dir ${TMDIR}/fb --patient_ids ${PIDS} --n_day_train 2 --n_day_test 1
python experiment/real_world/treatment/report_gprpp_tll.py --patient_ids ${PIDS} --model_dir ${TMDIR}
python evaluate/predict_figure3.py --patient_id 8 --remove_night_time --figure_dir out/figures/figure3 --plot_functional_components ao,b --treatment_model_dir ${TMDIR} --outcome_model_dir ${OMDIR}
python evaluate/predict_figure3.py --patient_id 8 --remove_night_time --figure_dir out/figures/figure7 --plot_functional_components bao,ba,bo,ao,b --treatment_model_dir ${TMDIR} --outcome_model_dir ${OMDIR}
python evaluate/plot_figure3.py --patient_id 8 --remove_night_time --figure_dir out/figures/figure3 --plot_functional_components ao,b --treatment_model_dir ${TMDIR} --outcome_model_dir ${OMDIR}
python evaluate/plot_figure3.py --patient_id 8 --remove_night_time --figure_dir out/figures/figure7 --plot_functional_components bao,ba,bo,ao,b --treatment_model_dir ${TMDIR} --outcome_model_dir ${OMDIR}
