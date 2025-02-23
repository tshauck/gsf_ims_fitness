"""
Stan utility functions for fitness model fitting
"""

import cmdstanpy
import pickle
import numpy as np
import pandas as pd
import os
import glob


def check_all_diagnostics(fit):
    print(fit.diagnose())


def file_to_list(file_name):
    text_file = open(file_name, "r")
    lines = text_file.readlines()
    return lines


def compile_model(
    filename,
    file_in_repository_models=True,
    check_includes=True,
    incl_stan_save_file=None,
):
    return_directory = os.getcwd()
    if file_in_repository_models:
        os.chdir(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "Stan models")
        )

    lines = file_to_list(filename)

    has_includes = False
    stan_file = filename
    while check_includes:
        check_includes = False
        new_lines = []
        for line in lines:
            if line.strip().startswith("#include"):
                include_file = line[line.find("#include") + 9 : line.rfind(".stan") + 5]
                include_file = include_file.strip()
                include_lines = file_to_list(include_file)
                new_lines += include_lines
                check_includes = True
                has_includes = True
            else:
                new_lines += [line]
        lines = new_lines

    if has_includes:
        if incl_stan_save_file is not None:
            os.chdir(return_directory)
            stan_file = incl_stan_save_file
        else:
            stan_file = filename.replace(".stan", ".incl.stan")

        with open(stan_file, "w") as out_file:
            out_file.writelines(lines)

    sm = cmdstanpy.CmdStanModel(stan_file=stan_file)

    os.chdir(return_directory)

    return sm


def check_rhat_by_params(fit, rhat_cutoff, stan_parameters=None):
    df = fit.summary()
    if stan_parameters is not None:
        key_params = np.array(df.index)
        sel = []
        for p in key_params:
            s = False
            for p2 in stan_parameters:
                if p2 in p:
                    s = True
                    break
            sel.append(s)
        key_params = key_params[sel]

        df = df.loc[key_params]

    df = df[df.R_hat > rhat_cutoff]

    return list(df.index)


def rhat_from_dataframe(df, split_chains=True):
    df = df.copy()
    if split_chains and ("draw" in list(df.columns)):
        chains = np.unique(df.chain)
        add_chain = max(chains) + 1
        cut_draw = (df.draw.max() + 1) / 2
        new_chain = []
        new_draw = []
        for c, d in zip(df.chain, df.draw):
            if d >= cut_draw:
                new_chain.append(c + add_chain)
                new_draw.append(d - cut_draw)
            else:
                new_chain.append(c)
                new_draw.append(d)
        df["chain"] = new_chain
        df["draw"] = new_draw

    chains = np.unique(df.chain)
    num_chains = len(chains)
    num_samples = len(df) / num_chains

    columns = list(df.columns)
    ignore_columns = ["chain", "draw", "warmup"] + [
        x for x in columns if x[-2:] == "__"
    ]
    columns = [x for x in columns if x not in ignore_columns]

    chain_mean = [df[df.chain == n][columns].mean() for n in chains]

    chain_mean = pd.concat(chain_mean, axis=1)

    grand_mean = chain_mean.mean(axis=1)

    x3 = chain_mean.sub(grand_mean, axis=0) ** 2
    between_chains_var = num_samples / (num_chains - 1) * x3.sum(axis=1)

    within_chains_var = []
    for n in chains:
        d2 = df[df.chain == n]
        d2 = d2[columns]
        x = ((d2 - chain_mean[n]) ** 2).sum()
        within_chains_var.append(1 / (num_samples - 1) * x)

    within_chains_var = pd.concat(within_chains_var, axis=1).mean(axis=1)

    rh1 = ((num_samples - 1) / num_samples) * within_chains_var
    rh2 = between_chains_var / num_samples

    return np.sqrt((rh1 + rh2) / within_chains_var)


def ref_fit_correction(lig_conc, plasmid, ligand=None, spike_in=None):
    if plasmid == "pRamR":
        y = 1 - 0.25 * lig_conc / 500
    elif ((plasmid == "pVER") and (ligand == "ONPF")) or (
        (plasmid == "pCymR") and (ligand == "Per-OH")
    ):
        fit_dict = fitness_calibration_dict(plasmid=plasmid)
        y_0 = fit_dict[0][spike_in](ligand, 0)[0]
        y_conc = fit_dict[0][spike_in](ligand, lig_conc)[0]
        y = y_conc / y_0
    else:
        y = 1
    return y


def fitness_calibration_dict(plasmid="pVER", barseq_directory=None, is_on_aws=False):
    # Dictionary of dictionaries of 2-tuple of functions
    #     first key is antibiotic concentration
    #     second key is spike-in name
    #     each function has two arguments: the ligand and the ligand concentration
    #         return from function is 2-tuple: (spike-in fitness, uncertainty of spike-in fitness)
    # ***** units for fitness values are 10-fold per plate. *****
    #         So, fitness=1 means that the cells grow 10-fold over the time for one plate repeate cycle

    spike_in_fitness_dict = {}
    if plasmid == "pVER":
        tet_list = [0, 1.25, 10, 20]
        # Fitness for 0, 1.25 and 10 are from 2022-11-22_two-lig_two-sel_OD-test-5-plates,
        # Fitness for 20 is from 2019 data, rescaled to match older zero-tet from 2022-11-22
        # TODO: move fitness values for spike-ins to somewhere else (not hard coded)
        # old: fitness_dicts = [{"AO-B": 0.9637, "AO-E": 0.9666}, {"AO-B": 0.9587125, "AO-E": 0.9597825},
        # old:                  {"AO-B": 0.93045, "AO-E": 0.92115}, {"AO-B": 0.8972, "AO-E": 0.8757}]

        fitness_dicts = [
            {"AO-B": 0.9288, "AO-E": 0.9282},
            {"AO-B": 0.9199, "AO-E": 0.9244},
            {"AO-B": 0.9063, "AO-E": 0.9014},
            {"AO-B": 0.8972 * 0.9288 / 0.9637, "AO-E": 0.8757 * 0.9282 / 0.9666},
        ]

        # Tet = 0, "AO-B":
        def fit_function(lig, conc):
            if (lig == "IPTG") or (lig == "none"):
                return (0.92379, 0.00168)
            if lig == "ONPF":
                return (
                    0.92379 * (1 - 0.02933 * conc / 2000),
                    0.00168 + 0.00378 * conc / 2000,
                )

        dict_list = [{"AO-B": fit_function}]

        # Tet = 1.25, "AO-B":
        def fit_function(lig, conc):
            if (lig == "IPTG") or (lig == "none"):
                return (0.91817, 0.00232)
            if lig == "ONPF":
                return (
                    0.91817 * (1 - 0.02933 * conc / 2000),
                    0.00232 + 0.00314 * conc / 2000,
                )

        dict_list += [{"AO-B": fit_function}]

        # Tet = 10, "AO-B":
        def fit_function(lig, conc):
            if (lig == "IPTG") or (lig == "none"):
                return (0.90297, 0.00262)
            if lig == "ONPF":
                return (
                    0.90297 * (1 - 0.02202 * conc / 2000),
                    0.00262 + 0.00314 * conc / 2000,
                )

        dict_list += [{"AO-B": fit_function}]

        # Tet = 20, "AO-B":
        def fit_function(lig, conc):
            if (lig == "IPTG") or (lig == "none"):
                return (0.8972 * 0.9288 / 0.9637, 0.005)
            if lig == "ONPF":
                return (np.nan, np.nan)

        dict_list += [{"AO-B": fit_function}]

        # Tet = 0, "AO-E":
        def fit_function(lig, conc):
            if (lig == "IPTG") or (lig == "none"):
                return (0.92789, 0.00165)
            if lig == "ONPF":
                return (
                    0.92789 * (1 - 0.03881 * conc / 2000),
                    0.00165 + 0.00367 * conc / 2000,
                )

        dict_list[0]["AO-E"] = fit_function

        # Tet = 1.25, "AO-E":
        def fit_function(lig, conc):
            if (lig == "IPTG") or (lig == "none"):
                return (0.92518, 0.00225)
            if lig == "ONPF":
                return (
                    0.92518 * (1 - 0.03881 * conc / 2000),
                    0.00225 + 0.00307 * conc / 2000,
                )

        dict_list[1]["AO-E"] = fit_function

        # Tet = 10, "AO-E":
        def fit_function(lig, conc):
            if (lig == "IPTG") or (lig == "none"):
                return (0.89910, 0.00263)
            if lig == "ONPF":
                return (
                    0.89910 * (1 - 0.01583 * conc / 2000),
                    0.00263 + 0.00286 * conc / 2000,
                )

        dict_list[2]["AO-E"] = fit_function

        # Tet = 20, "AO-E":
        def fit_function(lig, conc):
            if (lig == "IPTG") or (lig == "none"):
                return (0.8757 * 0.9282 / 0.9666, 0.005)
            if lig == "ONPF":
                return (np.nan, np.nan)

        dict_list[3]["AO-E"] = fit_function

        for t, d in zip(tet_list, dict_list):
            spike_in_fitness_dict[t] = d

    elif plasmid == "pCymR":
        tet_list = [0, 5]
        # Fitness for 0, and 5 Tet are indistinguishable in plate reader data.
        #     results are from 2023-11-17_Per-OH_OD-test-5-plates
        # AO-09 looks like a decent always-on, but AO-10 looks like it is actaully an inverted sensor,
        #     so use RS-20 instead, which is an always-on phenotype
        # TODO: move fitness values for spike-ins to somewhere else (not hard coded)

        # Tet = 0, "AO-09":
        def fit_function(lig, conc):
            # fitness is quadratic in Per-OH concentration (and uncertainty is also approximately quadratic
            fitness_popt = [
                0.96023440345,
                -0.0006537943555499999,
                -1.2593978344390849e-06,
            ]
            uncertainty_popt = [3.57965178e-03, -3.62885112e-06, 3.26292766e-07]
            if lig == "Per-OH":
                fit_ret = (
                    fitness_popt[0] + fitness_popt[1] * conc + fitness_popt[2] * conc**2
                )
                fit_ret_err = (
                    uncertainty_popt[0]
                    + uncertainty_popt[1] * conc
                    + uncertainty_popt[2] * conc**2
                )
                return (fit_ret, fit_ret_err)
            else:
                return (fitness_popt[0], uncertainty_popt[0])

        dict_list = [{"AO-09": fit_function}]

        # Tet = 5, "AO-09":
        # No measureable difference between with and without Tet
        dict_list += [{"AO-09": fit_function}]

        # Tet = 0, "RS-20":
        def fit_function(lig, conc):
            # fitness is quadratic in Per-OH concentration (and uncertainty is also approximately quadratic
            fitness_popt = [0.96257526335, -0.00073499716683, -4.774547993411e-07]
            uncertainty_popt = [4.79398178e-03, 1.67402328e-05, 2.73392737e-07]
            if lig == "Per-OH":
                fit_ret = (
                    fitness_popt[0] + fitness_popt[1] * conc + fitness_popt[2] * conc**2
                )
                fit_ret_err = (
                    uncertainty_popt[0]
                    + uncertainty_popt[1] * conc
                    + uncertainty_popt[2] * conc**2
                )
                return (fit_ret, fit_ret_err)
            else:
                return (fitness_popt[0], uncertainty_popt[0])

        dict_list[0]["RS-20"] = fit_function

        # Tet = 5, "RS-20":
        # No measureable difference between with and without Tet
        dict_list[1]["RS-20"] = fit_function

        for t, d in zip(tet_list, dict_list):
            spike_in_fitness_dict[t] = d

    elif plasmid == "pRamR":
        zeo_list = [0, 200]
        # Fitness interpolating functions are from data with Hamilton programming error (mixed up some of the Tet vs. non-Tet wells).
        #     Based on sucessful results (quantitative comparison between BarSeq and cytometry dose-response curves), it doesn't matter.
        #     Probably because the always-on controls here express the Zeo resistance at a high level so they have the same growth rate for all Zeo concentrations used.
        return_directory = os.getcwd()
        if not is_on_aws:
            fitness_exp_id = "2023-01-27_three_inducers_OD-test-5-plates"
            os.chdir(barseq_directory)
            direct = os.getcwd()
            while direct[-4:] != "RamR":
                os.chdir("..")
                direct = os.getcwd()
            os.chdir(fitness_exp_id)

        fit_files = glob.glob("fitness_vs_ligand_pRamR*.pkl")
        keys = [x[x.find("ON") : -4] for x in fit_files]
        values = [pickle.load(open(f, "rb")) for f in fit_files]
        os.chdir(return_directory)

        fitness_dicts = [dict(zip(keys, values)), dict(zip(keys, values))]

        for t, d in zip(zeo_list, fitness_dicts):
            spike_in_fitness_dict[t] = d

    elif plasmid == "Align-TF":
        """
        tet_list = [0, 0.5, 1, 5]
        # Fitness values are from 2024-08-27_Align-TF_GBA_1_OD-test,
        # TODO: move fitness values for spike-ins to somewhere else (not hard coded)

        # TMP = 0, "pRamR-norm-01":
        def fit_function(lig, conc):
            return (0.91051, 0.013531)
        dict_list = [{"pRamR-norm-01":fit_function}]

        # TMP = 0.5, "pRamR-norm-01":
        def fit_function(lig, conc):
            return (0.89862, 0.013568)
        dict_list += [{"pRamR-norm-01":fit_function}]

        # TMP = 1, "pRamR-norm-01":
        def fit_function(lig, conc):
            return (0.89142, 0.011612)
        dict_list += [{"pRamR-norm-01":fit_function}]

        # TMP = 5, "pRamR-norm-01":
        def fit_function(lig, conc):
            return (0.70381, 0.012593)
        dict_list += [{"pRamR-norm-01":fit_function}]

        # TMP = 0, "pLacI-norm-01":
        def fit_function(lig, conc):
            return (0.94828, 0.014503)
        dict_list[0]["pLacI-norm-01"] = fit_function

        # TMP = 0.5, "pLacI-norm-01":
        def fit_function(lig, conc):
            return (0.90892, 0.015545)
        dict_list[1]["pLacI-norm-01"] = fit_function

        # TMP = 1, "pLacI-norm-01":
        def fit_function(lig, conc):
            return (0.88796, 0.021021)
        dict_list[2]["pLacI-norm-01"] = fit_function

        # TMP = 5, "pLacI-norm-01":
        def fit_function(lig, conc):
            return (0.64319, 0.020423)
        dict_list[3]["pLacI-norm-01"] = fit_function
        """

        tmp_list = [0, 0.3, 1, 3]
        # Fitness values are from 2024-11-22_Align-TF_GBA_1_OD-test,
        # TODO: move fitness values for spike-ins to somewhere else (not hard coded)

        # "pRamR-norm-02", does not depend on [TMP]:
        def fit_function(lig, conc):
            if lig == "1S-TIQ":
                return (0.9795 - 0.0328 * conc / 250, 0.0094 + 0.0025 * conc / 250)
            else:
                return (0.9795, 0.0094)

        dict_list = [{"pRamR-norm-02": fit_function}] * 4

        # "pLacI-norm-02", does not depend on [TMP]:
        def fit_function(lig, conc):
            if lig == "1S-TIQ":
                return (0.9767 - 0.0328 * conc / 250, 0.0096 + 0.0025 * conc / 250)
            else:
                return (0.9767, 0.0096)

        for d in dict_list:
            d["pLacI-norm-02"] = fit_function

        for t, d in zip(tmp_list, dict_list):
            spike_in_fitness_dict[t] = d

    return spike_in_fitness_dict


def log_g_limits(plasmid="pVER"):
    if plasmid == "pVER":
        log_g_min = 0.5
        log_g_max = 4.7
        log_g_prior_scale = 0.15
        wild_type_ginf = 2.44697108e04
    elif plasmid == "pRamR":
        log_g_min = np.log10(2)
        log_g_max = 5
        log_g_prior_scale = 0.15
        wild_type_ginf = 10**4.67
    elif plasmid == "pCymR":
        log_g_min = np.log10(0.3)
        log_g_max = np.log10(500)
        log_g_prior_scale = 0.15
        wild_type_ginf = 10**2.481
    elif plasmid == "Align-TF":
        log_g_min = np.log10(5)
        log_g_max = np.log10(50000)
        log_g_prior_scale = np.nan
        wild_type_ginf = np.nan
    else:
        log_g_min = 1
        log_g_max = 4.5
        log_g_prior_scale = 0.3
        wild_type_ginf = 1839

    return (log_g_min, log_g_max, log_g_prior_scale, wild_type_ginf)


def get_stan_data(
    st_row,
    plot_df,
    antibiotic_conc_list,
    lig_list,
    fit_fitness_difference_params,
    old_style_columns=False,
    initial="b",
    plasmid="pVER",
    is_gp_model=False,
    min_err=0.05,
    ref_samples=None,
    apply_ramr_correction=True,
    ramr_fitness_correction=None,
    ramr_fitness_correction_params=None,
):
    log_g_min, log_g_max, log_g_prior_scale, _ = log_g_limits(plasmid=plasmid)

    antibiotic_conc_list = np.array(antibiotic_conc_list)

    spike_in = get_spike_in_name_from_inital(plasmid, initial)

    if old_style_columns:
        high_tet = antibiotic_conc_list[1]

        y_zero = st_row[f"fitness_{0}_estimate_{initial}"]
        s_zero = st_row[f"fitness_{0}_err_{initial}"]
        y_high = st_row[f"fitness_{high_tet}_estimate_{initial}"]
        s_high = st_row[f"fitness_{high_tet}_err_{initial}"]

        y = (y_high - y_zero) / y_zero
        s = np.sqrt(s_high**2 + s_zero**2) / y_zero

        x_fit = x
        x_y_s_list = [[x_fit, y, s]]
    else:
        y = [st_row[f"fitness_S{i}_{initial}"] for i in ref_samples]
        s = [st_row[f"fitness_S{i}_err_{initial}"] for i in ref_samples]
        y_ref_list = np.array(y)
        s_ref_list = np.array(s)

        sel = ~np.isnan(y_ref_list)
        if len(sel[sel]) > 0:
            y_ref_list = y_ref_list[sel]
            s_ref_list = s_ref_list[sel]

        w = 1 / s_ref_list**2
        y_ref = np.average(y_ref_list, weights=w)
        s_ref = np.average((y_ref_list - y_ref) ** 2, weights=w)
        v_1 = np.sum(w)
        v_2 = np.sum(w**2)
        s_ref = np.sqrt(s_ref / (1 - (v_2 / v_1**2)))

        tet_list = antibiotic_conc_list[antibiotic_conc_list > 0]
        x_y_s_list = []
        for lig in lig_list:
            sub_list = []
            for tet in tet_list:
                df = plot_df
                df = df[(df.ligand == lig) | (df.ligand == "none")]
                df = df[df.antibiotic_conc == tet]
                df = df.sort_values(by=lig)
                x = np.array(df[lig])
                samples = np.array(df.sample_id)
                # Correction factor for non-constant ref fitness (i.e., fitness decreases with [ligand]
                ref_correction = np.array(
                    [
                        ref_fit_correction(z, plasmid, ligand=lig, spike_in=spike_in)
                        for z in x
                    ]
                )
                y = np.array([st_row[f"fitness_S{i}_{initial}"] for i in df.sample_id])
                s = np.array(
                    [st_row[f"fitness_S{i}_err_{initial}"] for i in df.sample_id]
                )

                if plasmid in ["pVER", "pCymR"]:
                    y = (y - y_ref * ref_correction) / (y_ref * ref_correction)
                    s = np.sqrt(s**2 + (s_ref * ref_correction) ** 2) / (
                        y_ref * ref_correction
                    )
                if plasmid == "pRamR":
                    y = (y - y_ref) / (y_ref * ref_correction)
                    s = np.sqrt(s**2 + s_ref**2) / (y_ref * ref_correction)

                    # Ligand effects on fitness make the measurements at the highest concentration less reliable
                    s[x >= 500] *= 2

                    if apply_ramr_correction:
                        # Calibration correction for RamR system
                        ramr_model = ramr_fitness_correction
                        if ramr_model is None:
                            raise Exception(
                                "RamR calibration correction model (ramr_fitness_correction) is None"
                            )

                        params = ramr_fitness_correction_params
                        if ramr_model is None:
                            raise Exception(
                                "RamR calibration correction parameters (ramr_fitness_correction_params) is None"
                            )

                        df = df.copy()
                        df["lig_conc"] = x
                        df["fitness_effect"] = y
                        df["ref_fitness"] = [y_ref] * len(x)
                        df["early_fitness"] = np.array(
                            [st_row[f"fitness_S{i}_ea.{initial}"] for i in df.sample_id]
                        )

                        X_test = df[params]
                        X_test = X_test.dropna()

                        y_corr = ramr_model.predict(X_test)
                        y = y - y_corr

                s = np.sqrt(s**2 + min_err**2)

                if is_gp_model:
                    # For GP model, can't have missing data. So, if either y or s is nan, replace with values that won't affect GP model results (i.e. s=100)
                    invalid = np.isnan(y) | np.isnan(s)
                    if len(tet_list) == 1:
                        middle_fitness = fit_fitness_difference_params[0][0] / 2
                    elif len(tet_list) == 2:
                        middle_fitness = (
                            fit_fitness_difference_params[0][0]
                            + fit_fitness_difference_params[1][0]
                        ) / 4
                    y[invalid] = middle_fitness
                    s[invalid] = 100
                else:
                    valid = ~(np.isnan(y) | np.isnan(s))
                    x = x[valid]
                    y = y[valid]
                    s = s[valid]
                    samples = samples[valid]

                sub_list.append([x, y, s, samples])

            x_y_s_list.append(sub_list)

        if len(lig_list) == 1:
            # Case for single ligand and single antibiotic concentration
            if fit_fitness_difference_params is None:
                fit_fitness_difference_params = np.full((1, 6), np.nan)

            low_fitness = fit_fitness_difference_params[0][0]
            mid_g = fit_fitness_difference_params[0][1]
            fitness_n = fit_fitness_difference_params[0][2]

            x = x_y_s_list[0][0][0]
            y = x_y_s_list[0][0][1]
            y_err = x_y_s_list[0][0][2]
            samp = x_y_s_list[0][0][3]

            stan_data = dict(
                x=x,
                y=y,
                N=len(y),
                y_err=y_err,
                low_fitness_mu=low_fitness,
                mid_g_mu=mid_g,
                fitness_n_mu=fitness_n,
                log_g_min=log_g_min,
                log_g_max=log_g_max,
                log_g_prior_scale=log_g_prior_scale,
                y_ref=y_ref,
                samp=samp,
            )

        elif (len(lig_list) == 2) and (len(tet_list) == 2):
            # Case for two-tet, two-ligand (e.g., LacI with high and low tet)
            if fit_fitness_difference_params is None:
                fit_fitness_difference_params = np.full((2, 6), np.nan)

            y_0_med = x_y_s_list[0][0][1][0]
            s_0_med = x_y_s_list[0][0][2][0]
            samp_0_med = x_y_s_list[0][0][3][0]

            x_1 = x_y_s_list[0][0][0]
            y_1_med = x_y_s_list[0][0][1]
            s_1_med = x_y_s_list[0][0][2]
            samp_1_med = x_y_s_list[0][0][3]
            y_1_med = y_1_med[x_1 > 0]
            s_1_med = s_1_med[x_1 > 0]
            samp_1_med = samp_1_med[x_1 > 0]
            x_1 = x_1[x_1 > 0]

            x_1_high = x_y_s_list[0][1][0]
            y_1_high = x_y_s_list[0][1][1]
            s_1_high = x_y_s_list[0][1][2]
            samp_1_high = x_y_s_list[0][1][3]
            y_1_high = y_1_high[x_1_high > 0]
            s_1_high = s_1_high[x_1_high > 0]
            samp_1_high = samp_1_high[x_1_high > 0]
            x_1_high = x_1_high[x_1_high > 0]

            x_2 = x_y_s_list[1][0][0]
            y_2_med = x_y_s_list[1][0][1]
            s_2_med = x_y_s_list[1][0][2]
            samp_2_med = x_y_s_list[1][0][3]
            y_2_med = y_2_med[x_2 > 0]
            s_2_med = s_2_med[x_2 > 0]
            samp_2_med = samp_2_med[x_2 > 0]
            x_2 = x_2[x_2 > 0]

            x_2_high = x_y_s_list[1][1][0]
            y_2_high = x_y_s_list[1][1][1]
            s_2_high = x_y_s_list[1][1][2]
            samp_2_high = x_y_s_list[1][1][3]
            y_2_high = y_2_high[x_2_high > 0]
            s_2_high = s_2_high[x_2_high > 0]
            samp_2_high = samp_2_high[x_2_high > 0]
            x_2_high = x_2_high[x_2_high > 0]

            stan_data = dict(
                N_lig=len(x_1),
                x_1=x_1,
                x_2=x_2,
                y_0_low_tet=y_0_med,
                y_0_low_tet_err=s_0_med,
                y_1_low_tet=y_1_med,
                y_1_low_tet_err=s_1_med,
                y_2_low_tet=y_2_med,
                y_2_low_tet_err=s_2_med,
                y_1_high_tet=y_1_high,
                y_1_high_tet_err=s_1_high,
                y_2_high_tet=y_2_high,
                y_2_high_tet_err=s_2_high,
                log_g_min=log_g_min,
                log_g_max=log_g_max,
                log_g_prior_scale=log_g_prior_scale,
                low_fitness_mu_low_tet=fit_fitness_difference_params[0][0],
                mid_g_mu_low_tet=fit_fitness_difference_params[0][1],
                fitness_n_mu_low_tet=fit_fitness_difference_params[0][2],
                low_fitness_std_low_tet=fit_fitness_difference_params[0][3],
                mid_g_std_low_tet=fit_fitness_difference_params[0][4],
                fitness_n_std_low_tet=fit_fitness_difference_params[0][5],
                low_fitness_mu_high_tet=fit_fitness_difference_params[1][0],
                mid_g_mu_high_tet=fit_fitness_difference_params[1][1],
                fitness_n_mu_high_tet=fit_fitness_difference_params[1][2],
                low_fitness_std_high_tet=fit_fitness_difference_params[1][3],
                mid_g_std_high_tet=fit_fitness_difference_params[1][4],
                fitness_n_std_high_tet=fit_fitness_difference_params[1][5],
                y_ref=y_ref,
                samp_0_low_tet=samp_0_med,
                samp_1_low_tet=samp_1_med,
                samp_2_low_tet=samp_2_med,
                samp_1_high_tet=samp_1_high,
                samp_2_high_tet=samp_2_high,
            )

        elif (len(lig_list) == 3) and (len(tet_list) == 1):
            # Case for three-ligand experiment (e.g., RamR)
            # x_y_s_list: 1st index is the ligand (0, 1, or 2)
            #             2nd index is the antibiotic concentration (always 0 here)
            #             3rd index is 0 for x, 1 for y, 2 for s
            #             4th index is for individual data points
            if fit_fitness_difference_params is None:
                fit_fitness_difference_params = np.full((1, 6), np.nan)

            x_1, y_1, s_1, samp_1 = tuple(x_y_s_list[0][0][n] for n in range(4))
            y_0 = y_1[x_1 == 0]
            s_0 = s_1[x_1 == 0]
            samp_0 = samp_1[x_1 == 0]

            y_1 = y_1[x_1 > 0]
            s_1 = s_1[x_1 > 0]
            samp_1 = samp_1[x_1 > 0]
            x_1 = x_1[x_1 > 0]

            x_2, y_2, s_2, samp_2 = tuple(x_y_s_list[1][0][n] for n in range(4))
            y_2 = y_2[x_2 > 0]
            s_2 = s_2[x_2 > 0]
            samp_2 = samp_2[x_2 > 0]
            x_2 = x_2[x_2 > 0]

            x_3, y_3, s_3, samp_3 = tuple(x_y_s_list[2][0][n] for n in range(4))
            y_3 = y_3[x_3 > 0]
            s_3 = s_3[x_3 > 0]
            samp_3 = samp_3[x_3 > 0]
            x_3 = x_3[x_3 > 0]

            stan_data = dict(
                N_lig=len(x_1),
                y_0=y_0,
                y_0_err=s_0,
                x_1=x_1,
                y_1=y_1,
                y_1_err=s_1,
                x_2=x_2,
                y_2=y_2,
                y_2_err=s_2,
                x_3=x_3,
                y_3=y_3,
                y_3_err=s_3,
                samp_0=samp_0,
                samp_1=samp_1,
                samp_2=samp_2,
                samp_3=samp_3,
                log_g_min=log_g_min,
                log_g_max=log_g_max,
                log_g_prior_scale=log_g_prior_scale,
                mid_g_mu=fit_fitness_difference_params[0][1],
                fitness_n_mu=fit_fitness_difference_params[0][2],
                mid_g_std=fit_fitness_difference_params[0][4],
                fitness_n_std=fit_fitness_difference_params[0][5],
                y_ref=y_ref,
            )
            if plasmid == "pRamR":
                stan_data["high_fitness_mu"] = fit_fitness_difference_params[0][0]
                stan_data["high_fitness_std"] = fit_fitness_difference_params[0][3]
            else:
                stan_data["low_fitness_mu"] = fit_fitness_difference_params[0][0]
                stan_data["low_fitness_std"] = fit_fitness_difference_params[0][3]

    return stan_data


def get_spike_in_name_from_inital(plasmid, initial):
    if plasmid == "pVER":
        if initial[-1] == "b":
            spike_in = "AO-B"
        elif initial[-1] == "e":
            spike_in = "AO-E"
        else:
            raise ValueError(f"spike-in initial not recognized: {initial}")
    elif plasmid == "pRamR":
        if initial[-4:] == "sp01":
            spike_in = "ON-01"
        elif initial[-4:] == "sp02":
            spike_in = "ON-02"
        else:
            raise ValueError(f"spike-in initial not recognized: {initial}")
    elif plasmid == "pCymR":
        if initial[-4:] == "sp09":
            spike_in = "AO-09"
        elif initial[-4:] == "rs20":
            spike_in = "RS-20"
        else:
            raise ValueError(f"spike-in initial not recognized: {initial}")
    elif plasmid == "Align-TF":
        if initial[-4:] == "laci":
            spike_in = "pLacI-norm-02"
        elif initial[-4:] == "ramr":
            spike_in = "pRamR-norm-02"
        else:
            raise ValueError(f"spike-in initial not recognized: {initial}")

    return spike_in


def init_stan_GP_fit(fit_fitness_difference_params, single_tet, plasmid="pVER"):
    sig = np.random.uniform(1, 3)
    rho = np.random.uniform(0.9, 1.1)
    alpha = np.random.uniform(0.009, 0.011)

    if plasmid == "pVER":
        if single_tet:
            low_fitness = fit_fitness_difference_params[0][0]
            mid_g = fit_fitness_difference_params[0][1]
            fitness_n = fit_fitness_difference_params[0][2]

            return dict(
                sigma=sig,
                low_fitness=low_fitness,
                mid_g=mid_g,
                fitness_n=fitness_n,
                rho=rho,
                alpha=alpha,
            )
        else:
            return dict(
                sigma=sig,
                rho=rho,
                alpha=alpha,
                low_fitness_low_tet=fit_fitness_difference_params[0][0],
                mid_g_low_tet=fit_fitness_difference_params[0][1],
                fitness_n_low_tet=fit_fitness_difference_params[0][2],
                low_fitness_high_tet=fit_fitness_difference_params[1][0],
                mid_g_high_tet=fit_fitness_difference_params[1][1],
                fitness_n_high_tet=fit_fitness_difference_params[1][2],
            )
    elif plasmid == "pRamR":
        return dict(
            sigma=sig,
            rho=rho,
            alpha=alpha,
            high_fitness=fit_fitness_difference_params[0][0],
            mid_g=fit_fitness_difference_params[0][1],
            fitness_n=fit_fitness_difference_params[0][2],
        )
    elif plasmid == "pCymR":
        return dict(
            sigma=sig,
            rho=rho,
            alpha=alpha,
            low_fitness=fit_fitness_difference_params[0][0],
            mid_g=fit_fitness_difference_params[0][1],
            fitness_n=fit_fitness_difference_params[0][2],
        )


def init_stan_fit_single_ligand(stan_data, fit_fitness_difference_params):
    x_data = stan_data["x"]
    y_data = stan_data["y"]
    log_g0 = log_level(np.mean(y_data[:2]))
    log_ginf = log_level(np.mean(y_data[-2:]))

    min_ic = np.log10(min([i for i in x_data if i > 0]))
    max_ic = np.log10(max(x_data))
    log_ec50 = np.random.uniform(min_ic, max_ic)

    n = np.random.uniform(1.3, 1.7)

    sig = np.random.uniform(1, 3)

    low_fitness = fit_fitness_difference_params[0][0]
    mid_g = fit_fitness_difference_params[0][1]
    fitness_n = fit_fitness_difference_params[0][2]

    return dict(
        log_g0=log_g0,
        log_ginf=log_ginf,
        log_ec50=log_ec50,
        sensor_n=n,
        sigma=sig,
        low_fitness=low_fitness,
        mid_g=mid_g,
        fitness_n=fitness_n,
    )


def init_stan_fit_two_lig_two_tet(stan_data, fit_fitness_difference_params):
    min_ic = np.log10(min(stan_data["x_1"]))
    max_ic = np.log10(max(stan_data["x_1"]))
    log_ec50_1 = np.random.uniform(min_ic, max_ic)
    log_ec50_2 = np.random.uniform(min_ic, max_ic)

    n_1 = np.random.uniform(1.3, 1.7)
    n_2 = np.random.uniform(1.3, 1.7)

    sig = np.random.uniform(1, 3)

    # Indices for x_y_s_list[ligand][tet][x,y,s][n]
    return dict(
        log_g0=log_level(stan_data["y_0_low_tet"]),
        log_ginf_1=log_level(np.mean(stan_data["y_1_high_tet"][-2:])),
        log_ginf_2=log_level(np.mean(stan_data["y_2_high_tet"][-2:])),
        log_ec50_1=log_ec50_1,
        log_ec50_2=log_ec50_2,
        sensor_n_1=n_1,
        sensor_n_2=n_2,
        sigma=sig,
        low_fitness_low_tet=fit_fitness_difference_params[0][0],
        mid_g_low_tet=fit_fitness_difference_params[0][1],
        fitness_n_low_tet=fit_fitness_difference_params[0][2],
        low_fitness_high_tet=fit_fitness_difference_params[1][0],
        mid_g_high_tet=fit_fitness_difference_params[1][1],
        fitness_n_high_tet=fit_fitness_difference_params[1][2],
    )


def init_stan_fit_three_ligand(
    stan_data, fit_fitness_difference_params, plasmid="pRamR"
):
    min_ic = np.log10(min(stan_data["x_1"]))
    max_ic = np.log10(max(stan_data["x_1"]))
    log_ec50_1 = np.random.uniform(min_ic, max_ic)
    log_ec50_2 = np.random.uniform(min_ic, max_ic)
    log_ec50_3 = np.random.uniform(min_ic, max_ic)

    n_1 = np.random.uniform(1.3, 1.7)
    n_2 = np.random.uniform(1.3, 1.7)
    n_3 = np.random.uniform(1.3, 1.7)

    sig = np.random.uniform(1, 3)

    # Indices for x_y_s_list[ligand][tet][x,y,s][n]
    ret_dict = dict(
        log_g0=log_level(np.mean(stan_data["y_0"]), plasmid=plasmid),
        log_ginf_1=log_level(np.mean(stan_data["y_1"][-2:]), plasmid=plasmid),
        log_ginf_2=log_level(np.mean(stan_data["y_2"][-2:]), plasmid=plasmid),
        log_ginf_3=log_level(np.mean(stan_data["y_3"][-2:]), plasmid=plasmid),
        log_ec50_1=log_ec50_1,
        log_ec50_2=log_ec50_2,
        log_ec50_3=log_ec50_3,
        sensor_n_1=n_1,
        sensor_n_2=n_2,
        sensor_n_3=n_3,
        sigma=sig,
        mid_g=fit_fitness_difference_params[0][1],
        fitness_n=fit_fitness_difference_params[0][2],
    )
    if plasmid == "pRamR":
        ret_dict["high_fitness"] = fit_fitness_difference_params[0][0]
    else:
        ret_dict["low_fitness"] = fit_fitness_difference_params[0][0]
    return ret_dict


def init_stan_fit_single_point(stan_data):
    sig = np.random.uniform(1, 3)

    return dict(
        sigma=sig,
        low_fitness=stan_data["low_fitness_mu"],
        mid_g=stan_data["mid_g_mu"],
        fitness_n=stan_data["fitness_n_mu"],
    )


def log_level(fitness_difference, plasmid="pVER"):
    if plasmid == "pVER":
        log_g = 1.439 * fitness_difference + 3.32
        log_g = log_g * np.random.uniform(0.9, 1.1)
        if log_g < 1.5:
            log_g = 1.5
        if log_g > 4:
            log_g = 4
        return log_g
    elif plasmid == "pRamR":
        log_g = -2.1 * fitness_difference / 1.5 + 2
        log_g = log_g * np.random.uniform(0.9, 1.1)
        if log_g < 2:
            log_g = 2
        if log_g > 4.5:
            log_g = 4.5
        return log_g
    elif plasmid == "pCymR":
        log_g = np.log10(200) * (1 + fitness_difference)
        log_g = log_g * np.random.uniform(0.9, 1.1)
        if log_g < 0:
            log_g = 0
        if log_g > np.log10(300):
            log_g = np.log10(300)
        return log_g

    raise ValueError(f"Unexpected plasmid: {plasmid}")
