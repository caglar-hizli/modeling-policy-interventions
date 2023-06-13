# dep_packages <- c("Rcpp","RcppArmadillo", "RcppEigen", "RcppNumerical", "MASS", "pracma", "mvtnorm", "LaplacesDemon","stats", "GoFKernel","survival","reticulate")
# new.packages <- dep_packages[!(dep_packages %in% installed.packages()[,"Package"])]
# if(length(new.packages)) install.packages(new.packages)
# install.packages("doct_1.0.tar.gz", repos = NULL, type="source")

# constants
n_patient = 50  # number of patients
n_train_day = 1  # number of training days
action_ids = "4,13"  # action patient ids
outcome_ids = "3,8,12" # outcome patient ids
n_outcome = 40  # number of outcome measurements

library(reticulate)
library(doct)
np <- import("numpy")
args = commandArgs(trailingOnly=TRUE)
sample_path = args[1]
seed = args[2]

# seed=1
# for (seed in 1:10) {
main_folder <- paste(sample_path, "/joint.seed", seed, sep='')
run_id <- paste("/np", n_patient, ".ntr", n_train_day, ".pa", action_ids, ".po", outcome_ids, ".no", n_outcome, ".tTrue", sep='')
npz0 <- np$load(paste(main_folder, run_id, "/treatment_data_p0_s", seed, ".npz", sep=''))
npz1 <- np$load(paste(main_folder, run_id, "/treatment_data_p1_s", seed, ".npz", sep=''))
npz2 <- np$load(paste(main_folder, run_id, "/outcome_data_s", seed, ".npz", sep=''))

simulated_treatment0 <- list(D=npz0$f[['D']], Y=npz0$f[['Y']], Ts=npz0$f[['Ts']], id=npz0$f[['id']],
                           censor=npz0$f[['censor']], surv_time=npz0$f[['surv_time']], Npat=npz0$f[['Npat']])
simulated_treatment1 <- list(D=npz1$f[['D']], Y=npz1$f[['Y']], Ts=npz1$f[['Ts']], id=npz1$f[['id']],
                           censor=npz1$f[['censor']], surv_time=npz1$f[['surv_time']], Npat=npz1$f[['Npat']])
simulated_outcome <- list(D=npz2$f[['D']], Y=npz2$f[['Y']], Ts=npz2$f[['Ts']], id=npz2$f[['id']],
                          censor=npz2$f[['censor']], surv_time=npz2$f[['surv_time']], Npat=npz2$f[['Npat']])

mcmc_settings=NULL
mcmc_settings$Niter=20000
mcmc_settings$burn.in=5000
mcmc_settings$ndisplay=500
mcmc_settings$peak_dist='gamma'
thin=50
post_thin_iters<-seq(mcmc_settings$burn.in+thin,mcmc_settings$Niter,thin)
set.seed(seed)
mcmc_treatment0<-mcmc_separate_treatment(simulated_treatment0,mcmc_settings,seed)
np$savez(paste(sample_path, "/mcmc_output/mcmc_treatment0_i20000_s", seed, ".npz", sep=''), beta_v=mcmc_treatment0$beta_v, theta_a=mcmc_treatment0$theta_a, mu=mcmc_treatment0$mu, k=mcmc_treatment0$k, Y=simulated_treatment0$Y, Ts=simulated_treatment0$Ts, Npat=simulated_treatment0$Npat, thin_ids=post_thin_iters)

mcmc_treatment1<-mcmc_separate_treatment(simulated_treatment1,mcmc_settings,seed)
np$savez(paste(sample_path, "/mcmc_output/mcmc_treatment1_i20000_s", seed, ".npz", sep=''), beta_v=mcmc_treatment1$beta_v, theta_a=mcmc_treatment1$theta_a, mu=mcmc_treatment1$mu, k=mcmc_treatment1$k, Y=simulated_treatment1$Y, Ts=simulated_treatment1$Ts, Npat=simulated_treatment1$Npat, thin_ids=post_thin_iters)

mcmc_outcome<-mcmc_separate_outcome(simulated_outcome,mcmc_settings,seed)
np$savez(paste(sample_path, "/mcmc_output/mcmc_outcome_i20000_s", seed, ".npz", sep=''), beta_l=mcmc_outcome$beta_l, b_il=mcmc_outcome$b_il, Y=simulated_outcome$Y, Ts=simulated_outcome$Ts, Npat=simulated_outcome$Npat, thin_ids=post_thin_iters)



