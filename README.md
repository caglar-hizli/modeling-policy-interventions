# Causal Modeling of Policy Interventions from Treatment-Outcome Sequences

This repository contains the public code to reproduce experiments and figures of the paper ["Causal Modeling of Policy Interventions From Treatment-Outcome Sequences" by Hızlı et al., ICML 2023](https://proceedings.mlr.press/v202/hizli23a.html).

### Citation

If you build on top of this code, please cite
```bibtex
@InProceedings{hizli23a,
  title = 	 {Causal Modeling of Policy Interventions from Treatment--Outcome Sequences},
  author =       {H{\i}zl{\i}, \c{C}a\u{g}lar and John, S. T. and Juuti, Anne Tuulikki and Saarinen, Tuure Tapani and Pietil\"{a}inen, Kirsi Hannele and Marttinen, Pekka},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {13050--13084},
  year = 	 {2023},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  pdf = 	 {https://proceedings.mlr.press/v202/hizli23a/hizli23a.pdf},
  url = 	 {https://proceedings.mlr.press/v202/hizli23a.html},
}
```

## Development environment and dependencies

### Python

The code has been developed and tested with Python 3.9 under MacOS X. Under other operating systems, you have to adjust the `tensorflow` package in `requirements-with-deps.txt`.
After activating a python environment, run
```bash
pip install -r requirements-with-deps.txt
pip install -r requirements-without-deps.txt
```

### R

The baseline of Hua et al. (2021) requires R.
The code has been tested both with R v4.2.0 and v3.6.3.
It requires the locally provided `doct` package and `reticulate` from CRAN, which can be installed with the following code:
```R
install.packages("remote")
install_local("models/benchmarks/hua/doct/doct_1.0.tar.gz")
install.packages("reticulate")
```

## Reproducing experiments

To reproduce the experiments on real-world data (section 7.1), run
```bash
PYTHONPATH=. bash run_experiment_rw.sh
```
This will recreate
- Figure 4 in `out/figures/figure3/train_fit_d1_p8_fig3.pdf`
- Figure 7 in `out/figures/figure7/train_fit_d1_p8_fig7.pdf`
- Figure 8 in `out/sampler/outcome.p3,8,12/figures/f_all_fit_id{0,1,2}_vTrue.pdf`
- Figure 9 in `out/sampler/action_pred/compare_{ga,go,mark_intensity}.pdf`
- Figure 10 in `out/sampler/outcome.p3,8,12/figures/compare_{fb,ft_m3}.pdf`
- Figure 11:
  - (a) in `figures.figure11/fig11_real_data_p3_d0.pdf`
  - (b) in `figures.figure11/fig11_real_data_p8_d1.pdf`
  - (c) in `figures.figure11/fig11_synth_data_p3_s12,15,27.pdf`
  - (d) in `figures.figure11/fig11_synth_data_p8_s1,4,25.pdf`

To reproduce the experiments on semi-synthetic data (section 7.2), run
```bash
PYTHONPATH=. bash run_experiment_synth.sh
```
(this will fit to real-world data to run the sampler to generate the semi-synthetic data).

This will recreate
- Figure 12
- Figure 13

## Configuration

Hyperparameters and other settings are defined in `utils/constants_rw.py` for the real-world experiment (section 7.1) and in `utils/constants_synth.py` for the semi-synthetic experiment (section 7.2).
