import argparse
import os
import numpy as np


def get_tm_run_args(patient_id, output_dir, args):
    if args.action_components == 'bao':
        return get_tm_fao_run_args(patient_id, output_dir, args)
    elif args.action_components == 'ao':
        return get_tm_fao_run_args(patient_id, output_dir, args)
    elif args.action_components == 'a':
        return get_tm_fa_run_args(patient_id, output_dir, args)
    else:
        raise ValueError('Get run args function not implemented!')


def get_tm_fbao_run_args(patient_id, output_dir, args):
    treatment_args = argparse.Namespace(
        model_dir=output_dir,
        T_treatment=3.0,
        patient_ids=[patient_id],
        patient_id=patient_id,
        n_day_train=2,
        M_times=20,
        variance_init=[0.05, 0.15],
        lengthscales_init=[1.0, 100.0, 2.5],
        variance_prior=[0.05, 0.15],
        lengthscales_prior=[1.0, 100.0, 2.5],
        beta0=0.1,
        remove_night_time=args.remove_night_time,
        action_components=args.action_components,
        marked=True,
        Dt=2,
        marked_dt=[1],
        D=3,
        action_dim=0,
        outcome_dim=1,
        preprocess_treatments=False,
        domain=[0.0, 24.0],
        share_variance=False,
        share_lengthscales=False,
        optimize_variance=False,
        optimize_lengthscales=False,
        compute_nll=False,
        maxiter=args.maxiter,
        save_sampler=False,
        sampler_mode=True,
    )
    treatment_args.time_dims = list(range(treatment_args.Dt))
    treatment_args.mark_dims = list(range(treatment_args.Dt, treatment_args.D))
    return treatment_args


def get_tm_fao_run_args(patient_id, output_dir, args):
    treatment_args = argparse.Namespace(
        model_dir=output_dir,
        T_treatment=3.0,
        patient_ids=[patient_id],
        patient_id=patient_id,
        n_day_train=200,
        M_times=20,
        variance_init=[0.05, 0.15],
        lengthscales_init=[1.0, 100.0, 2.5],
        variance_prior=[0.05, 0.15],
        lengthscales_prior=[1.0, 100.0, 2.5],
        beta0=0.1,
        remove_night_time=args.remove_night_time,
        sampling_rate=1,
        action_components=args.action_components,
        marked=True,
        Dt=2,
        marked_dt=[1],
        D=3,
        action_dim=0,
        outcome_dim=1,
        preprocess_treatments=False,
        domain=[0.0, 24.0],
        share_variance=False,
        share_lengthscales=False,
        optimize_variance=False,
        optimize_lengthscales=False,
        compute_nll=False,
        maxiter=args.maxiter,
        save_sampler=False,
        sampler_mode=True,
    )
    treatment_args.time_dims = list(range(treatment_args.Dt))
    treatment_args.mark_dims = list(range(treatment_args.Dt, treatment_args.D))
    return treatment_args


def get_tm_fa_run_args(patient_id, output_dir, args):
    treatment_args = argparse.Namespace(
        model_dir=output_dir,
        T_treatment=3.0,
        patient_ids=[patient_id],
        patient_id=patient_id,
        n_day_train=200,
        M_times=20,
        variance_init=[0.5],
        lengthscales_init=[1.0],
        variance_prior=[0.5],
        lengthscales_prior=[1.0],
        beta0=0.1,
        remove_night_time=args.remove_night_time,
        sampling_rate=1,
        action_components=args.action_components,
        marked=False,
        Dt=1,
        marked_dt=[],
        D=1,
        action_dim=0,
        outcome_dim=-1,
        preprocess_treatments=False,
        domain=[0.0, 24.0],
        share_variance=False,
        share_lengthscales=False,
        optimize_variance=False,
        optimize_lengthscales=False,
        compute_nll=False,
        maxiter=args.maxiter,
        save_sampler=False,
        sampler_mode=True,
    )
    treatment_args.time_dims = list(range(treatment_args.Dt))
    treatment_args.mark_dims = list(range(treatment_args.Dt, treatment_args.D))
    return treatment_args
