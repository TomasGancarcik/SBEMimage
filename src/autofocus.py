# -*- coding: utf-8 -*-

# ==============================================================================
#   This source file is part of SBEMimage (github.com/SBEMimage)
#   (c) 2018-2020 Friedrich Miescher Institute for Biomedical Research, Basel,
#   and the SBEMimage developers.
#   This software is licensed under the terms of the MIT License.
#   See LICENSE.txt in the project root folder.
# ==============================================================================

"""
This module provides automatic focusing and stigmation. Two methods are
implemented: (1) SmartSEM autofocus, which is called in user-specified
intervals on selected tiles. (2) Heuristic algorithm as used in Briggman
et al. (2011), described in Appendix A of Binding et al. (2012).
"""

import os.path
import random
from time import sleep, time
from typing import Union, Tuple, Optional, List, Any
from math import sqrt, exp, sin, cos
from statistics import mean

import json
import skimage.io
import numpy as np
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

import autofocus_mapfost
import utils


class Autofocus():

    def __init__(self, config, sem, grid_manager):
        self.cfg = config
        self.sem = sem
        self.gm = grid_manager
        self.method = int(self.cfg['autofocus']['method'])
        self.tracking_mode = int(self.cfg['autofocus']['tracking_mode'])
        self.interval = int(self.cfg['autofocus']['interval'])
        self.autostig_delay = int(self.cfg['autofocus']['autostig_delay'])
        self.pixel_size = float(self.cfg['autofocus']['pixel_size'])
        # Maximum allowed change in focus/stigmation
        self.max_wd_diff, self.max_stig_x_diff, self.max_stig_y_diff = (
            json.loads(self.cfg['autofocus']['max_wd_stig_diff']))
        # For the heuristic autofocus method, a dictionary of cropped central
        # tile areas is kept in memory for processing during the cut cycles.
        self.img = {}
        self.wd_delta, self.stig_x_delta, self.stig_y_delta = json.loads(
            self.cfg['autofocus']['heuristic_deltas'])
        self.heuristic_calibration = json.loads(
            self.cfg['autofocus']['heuristic_calibration'])
        self.rot_angle, self.scale_factor = json.loads(
            self.cfg['autofocus']['heuristic_rot_scale'])
        self.ACORR_WIDTH = 64
        self.fi_mask = np.empty((self.ACORR_WIDTH, self.ACORR_WIDTH))
        self.fo_mask = np.empty((self.ACORR_WIDTH, self.ACORR_WIDTH))
        self.apx_mask = np.empty((self.ACORR_WIDTH, self.ACORR_WIDTH))
        self.amx_mask = np.empty((self.ACORR_WIDTH, self.ACORR_WIDTH))
        self.apy_mask = np.empty((self.ACORR_WIDTH, self.ACORR_WIDTH))
        self.amy_mask = np.empty((self.ACORR_WIDTH, self.ACORR_WIDTH))
        self.make_heuristic_weight_function_masks()
        # Estimators (dicts with tile keys)
        self.foc_est = {}
        self.astgx_est = {}
        self.astgy_est = {}
        # Computed corrections for heuristic autofocus
        self.wd_stig_corr = {}
        # If MagC mode active, enforce method and tracking mode
        self.magc_mode = (self.cfg['sys']['magc_mode'].lower() == 'true')
        if self.magc_mode:
            self.method = 0  # SmartSEM autofocus
            self.tracking_mode = 0  # Track selected, approx. others

        self.MAPFOST_PATCH_SIZE = [768, 768]
        self.MAPFOST_FRAME_RESOLUTION = 2
        self.mapfost_wd_pert = float(self.cfg['autofocus']['mapfost_wd_perturbations'])
        self.mapfost_dwell_time = float(self.cfg['autofocus']['mapfost_dwell_time'])
        self.mapfost_max_iters = int(self.cfg['autofocus']['mapfost_maximum_iterations'])
        self.mapfost_conv_thresh = float(self.cfg['autofocus']['mapfost_convergence_threshold_um'])
        self.mapfost_large_aberrations = int(self.cfg['autofocus']['mapfost_large_aberrations'])

        # Mapfost Calibration Parameters
        self.mapfost_probe_conv = float(self.cfg['autofocus']['mapfost_probe_convergence_angle'])
        self.mapfost_stig_rot = float(self.cfg['autofocus']['mapfost_astig_rotation_deg'])
        self.mapfost_stig_scale = json.loads(self.cfg['autofocus']['mapfost_astig_scaling'])

        # Automated Focus/Stigmator Series
        self.afss_wd_delta = json.loads(self.cfg['autofocus']['afss_wd_delta'])
        self.afss_stig_x_delta = json.loads(self.cfg['autofocus']['afss_stig_x_delta'])  # percent
        self.afss_stig_y_delta = json.loads(self.cfg['autofocus']['afss_stig_y_delta'])  # percent
        self.afss_rounds = json.loads(self.cfg['autofocus']['afss_rounds'])  # number of induced focus/stig deviations
        self.afss_offset = json.loads(
            self.cfg['autofocus']['afss_offset'])  # skip N slices before first AFSS activation
        self.afss_current_round = 0  # position of current WD/stig deviation within AFSS series
        self.afss_next_activation = 0  # slice nr of nearest planned AFSS run
        self.afss_perturbation_series = {}  # series that holds factors by which is the wd/stig delta multiplied
        self.afss_wd_stig_orig = {}  # original values before the AFSS started: d = {tile_keys:[[wd, dummy=0], (sx,sy)]}
        # dict = {tile_keys: {slice_nrs: [ (wd, dummy=0), (sx,sy), sharpness, img_full_path, stddev, [shift_vec] ]}}
        self.afss_wd_stig_corr = {}
        self.afss_wd_stig_corr_optima = {}  # Computed corrections AFSS: dict = {tile_keys: [wd/stig opt.val, fit_rmse]}
        self.afss_mode = self.cfg['autofocus']['afss_mode']  # 'focus' 'stig_x' 'stig_y'  # allows defining type of
        # afss series to be used at the beginning of acquisition
        self.afss_consensus_mode = int(self.cfg['autofocus']['afss_consensus_mode'])  # 0: 'Average', 1: 'Tile specific'
        # or 2: 'Focus (Specific), Stig (Average)
        self.afss_drift_corrected = (self.cfg['autofocus']['afss_drift_corrected'].lower() == 'true')
        self.afss_active = False  # this might be beneficial for implementing continuation of afss series after pause
        self.afss_interpolation_method = 'polyfit'  # fct to be used for interpolating the measured sharpness values
        self.afss_autostig_active = (self.cfg['autofocus']['afss_autostig_active'].lower() == 'true')
        self.afss_hyper_perturbation_series = {}
        self.afss_shuffle = True
        self.afss_hyper_shuffle = False
        self.afss_filter_outliers = True
        self.afss_weighted_averaging = True
        self.afss_avg_corr = None
        self.afss_max_fails = json.loads(self.cfg['autofocus']['afss_max_fails'])
        self.afss_rmse_limit = float(self.cfg['autofocus']['afss_rmse_limit'])

    def save_to_cfg(self):
        """Save current autofocus settings to ConfigParser object. Note that
        autofocus reference tiles are managed in grid_manager.
        """
        self.cfg['autofocus']['method'] = str(self.method)
        self.cfg['autofocus']['tracking_mode'] = str(self.tracking_mode)
        self.cfg['autofocus']['max_wd_stig_diff'] = str(
            [self.max_wd_diff, self.max_stig_x_diff, self.max_stig_y_diff])
        self.cfg['autofocus']['interval'] = str(self.interval)
        self.cfg['autofocus']['autostig_delay'] = str(self.autostig_delay)
        self.cfg['autofocus']['pixel_size'] = str(self.pixel_size)
        self.cfg['autofocus']['heuristic_deltas'] = str(
            [self.wd_delta, self.stig_x_delta, self.stig_y_delta])
        self.cfg['autofocus']['heuristic_calibration'] = str(
            self.heuristic_calibration)
        self.cfg['autofocus']['heuristic_rot_scale'] = str(
            [self.rot_angle, self.scale_factor])
        self.cfg['autofocus']['afss_wd_delta'] = str(self.afss_wd_delta)
        self.cfg['autofocus']['afss_stig_x_delta'] = str(self.afss_stig_x_delta)
        self.cfg['autofocus']['afss_stig_y_delta'] = str(self.afss_stig_y_delta)
        self.cfg['autofocus']['afss_rounds'] = str(self.afss_rounds)
        self.cfg['autofocus']['afss_offset'] = str(self.afss_offset)
        self.cfg['autofocus']['afss_consensus_mode'] = str(self.afss_consensus_mode)
        self.cfg['autofocus']['afss_drift_corrected'] = str(self.afss_drift_corrected)
        self.cfg['autofocus']['afss_autostig_active'] = str(self.afss_autostig_active)
        self.cfg['autofocus']['afss_mode'] = str(self.afss_mode)
        self.cfg['autofocus']['afss_max_fails'] = str(self.afss_max_fails)
        self.cfg['autofocus']['afss_rmse_limit'] = str(self.afss_rmse_limit)

    # ================ Below: methods for Automated focus/stig series method ==================

    def afss_verify_results(self) -> Tuple[int, dict, bool, dict]:
        m = self.afss_mode
        rmse_limit = 1.0e2
        rmse_limit = self.afss_rmse_limit

        rejected_fits = {}
        rejected_thr = {}
        thr_ok = True

        # Remove corrupted results from optima dict, due unsuccessful fit(s)
        d = self.afss_wd_stig_corr_optima
        for t, vals in list(d.items()):
            rmse_val = vals[1]
            if rmse_val > rmse_limit or rmse_val == -1:
                del d[t]
                msg = f'Tile {t} rejected. RMSE: {round(rmse_val, 4)}'
                rejected_fits[t] = (rmse_val, msg)
        nr_of_reliable_fits = len(d)

        # Check that computed optimal WD and Stigmator values of ALL ref. tiles are below WD/Stig thresholds
        d = {'focus': (0, 0, self.max_wd_diff, 10 ** 6, 'WD', 'um'),
             'stig_x': (1, 0, self.max_stig_x_diff, 1, 'StigX', '%'),
             'stig_y': (1, 1, self.max_stig_y_diff, 1, 'StigY', '%')}
        if nr_of_reliable_fits != 0:  # Continue only if there is non-zero amount of corrections left from step 1.
            for tile_key, opt in self.afss_wd_stig_corr_optima.items():
                if self.afss_avg_corr is not None:  # Averaging mode is active
                    diff = abs(self.afss_avg_corr)
                    if diff >= d[m][2]:  # average diff is out of range
                        d1, d2 = round(self.afss_avg_corr * d[m][3], 3), round(d[m][2] * d[m][3], 3)
                        msg = f'Average {d[m][4]} correction: {d1} {d[m][5]} (Limit: {d2} {d[m][5]})'
                        rejected_thr[tile_key] = (d1, msg)
                        thr_ok = False
                        break
                else:  # Non-averaging mode is active: check that every new optimal value fits in permitted range
                    diff = abs(opt[0] - self.afss_wd_stig_orig[tile_key][d[m][0]][d[m][1]])
                    if diff >= d[m][2]:
                        d1, d2 = round(diff * d[m][3], 3), round(d[m][2] * d[m][3], 3)
                        msg = f'Tile {tile_key}: diff{d[m][4]}: {d1} {d[m][5]}, limit: {d2} {d[m][5]}'
                        rejected_thr[tile_key] = (d1, msg)
                thr_ok &= diff <= d[m][2]
        return nr_of_reliable_fits, rejected_fits, thr_ok, rejected_thr

    def afss_compute_pair_drifts(self):
        for tile_key in self.afss_wd_stig_corr:
            # print(f'Processing drifts: {tile_key} ')
            filenames = []
            for slice_nr in self.afss_wd_stig_corr[tile_key]:
                img_path = self.afss_wd_stig_corr[tile_key][slice_nr][3]
                filenames.append(img_path)
            newest_img_pair_fns = filenames[-2:]
            ic = utils.load_image_collection(newest_img_pair_fns)
            shift_vec = utils.register_image_collection(ic)
            self.afss_wd_stig_corr[tile_key][slice_nr].append(shift_vec)

    def process_afss_collections(self):
        for tile_key in self.afss_wd_stig_corr:
            # print(f'Processing collection: {tile_key} ')
            filenames = []
            basenames = []
            shifts = []
            for i, slice_nr in enumerate(self.afss_wd_stig_corr[tile_key]):
                img_path = self.afss_wd_stig_corr[tile_key][slice_nr][3]
                filenames.append(img_path)
                basenames.append(os.path.basename(img_path))
                if i != 0:  # Skip reading shift vector of first image as this was not registered to anything
                    shifts.append(self.afss_wd_stig_corr[tile_key][slice_nr][5][0])
            cumm_shifts = np.cumsum(shifts, axis=0)
            ic = utils.load_image_collection(filenames)
            ic = utils.shift_collection(ic, cumm_shifts)
            ic = utils.crop_image_collection(ic, cumm_shifts)
            coll_sharpness = utils.get_collection_sharpness(ic, metric='edges')  # based on custom mask defined
            # by shape of images in the collection; mode: 'contrast', 'edges'

            # Fill the results' dict with sharpness values from drift-corrected image collection
            for i, slice_nr in enumerate(self.afss_wd_stig_corr[tile_key]):
                # print(f'Populating {tile_key}, slice_nr: {slice_nr} with sharpness value: {coll_sharpness[i]}\n')
                self.afss_wd_stig_corr[tile_key][slice_nr][2] = coll_sharpness[i]
                reg_img_path = os.path.join(self.cfg['acq']['base_dir'], 'meta', 'stats', basenames[i])
                skimage.io.imsave(reg_img_path, ic[i])

    def fit_afss_collections(self, plot_results=True):
        def norm_data(arr: np.ndarray) -> np.ndarray:
            arr -= np.min(arr)
            arr /= np.max(arr)
            return arr
        m = self.afss_mode
        # print(self.afss_wd_stig_corr)
        for tile_key in self.afss_wd_stig_corr:
            # print(f'Fitting collection: {tile_key}')
            tile_dict = self.afss_wd_stig_corr[tile_key]  # Values of particular tile to be processed
            x_vals = np.asarray([], dtype=float)
            y_vals = np.asarray([], dtype=float)
            y_vals_std = np.asarray([], dtype=float)

            # read the values (wd/stig_x/stig_y, sharpness)
            d = {'focus': (0, 0), 'stig_x': (1, 0), 'stig_y': (1, 1)}
            x_orig = self.afss_wd_stig_orig[tile_key][d[m][0]][d[m][1]]  # for plotting purposes
            for slice_nr in tile_dict:
                x_vals = np.append(x_vals, tile_dict[slice_nr][d[m][0]][d[m][1]])  # WD, StigX or StigY series
                y_vals = np.append(y_vals, tile_dict[slice_nr][2])  # List of sharpness values
                y_vals_std = np.append(y_vals_std, tile_dict[slice_nr][4])  # List of 'contrast' values
            y_vals = np.sqrt(norm_data(norm_data(y_vals)**2 + norm_data(y_vals_std)**2))  # Combined sharpness metric

            # INTERPOLATION
            if self.afss_interpolation_method == 'spline':
                # SPLINE INTERPOLATION
                f1 = interp1d(x_vals, y_vals, kind='cubic')
                x_fit = np.linspace(min(x_vals), max(x_vals), num=101, endpoint=True)
                y_fit = f1(x_fit)
                x_opt, y_opt = x_fit[np.argmax(y_fit)], max(y_fit)  # x,y coordinates of optimal value
                self.afss_wd_stig_corr_optima[tile_key] = [x_opt, 0]
            elif self.afss_interpolation_method == 'polyfit':
                # POLYNOMIAL FIT
                cfs = np.polyfit(x_vals, y_vals, 2)
                x_fit = np.linspace(min(x_vals), max(x_vals), num=1001, endpoint=True)
                y_fit = cfs[0] * x_fit ** 2 + cfs[1] * x_fit + cfs[2]
                x_opt = -cfs[1] / (2 * cfs[0])
                y_opt = cfs[0] * x_opt ** 2 + cfs[1] * x_opt + cfs[2]

                if cfs[0] < 0:  # sharpness values follow expected (negative) quadratic behavior
                    y_func = utils.return_func_vals(cfs, x_vals)
                    rmse_val = utils.rmse(y_func, y_vals)
                else:  # fit has bad 'orientation'
                    rmse_val = -1
                self.afss_wd_stig_corr_optima[tile_key] = [x_opt, rmse_val]

            # Save resulting plots into the 'meta/stats/' folder
            if plot_results:
                plot_path = self.generate_afss_plot_path(tile_key, interpolation=self.afss_interpolation_method)
                self.plot_afss_series(x_vals=np.asarray(x_vals), y_vals=np.asarray(y_vals),
                                      x_fit=x_fit, y_fit=y_fit,
                                      x_opt=x_opt, y_opt=y_opt,
                                      x_orig=x_orig, err=rmse_val,
                                      path=plot_path)
        if self.afss_consensus_mode == 0 or (self.afss_consensus_mode == 2 and self.afss_mode != 'focus'):
            _, __ = self.get_average_afss_correction(do_filtering=self.afss_filter_outliers,
                                                     do_weighted_average=self.afss_weighted_averaging)
        self.afss_wd_stig_corr = {}  # Reset the correction dictionary to prepare it for next afss run

    def generate_afss_plot_path(self, tile_key: str, interpolation: str) -> str:
        # Generate plot name: basedir + 'slice_nr'_'grid_nr'_'tile_nr'_'focus/x_stig/y_stig_fit'.png
        # Interpolation methods: 'spline', 'polyfit'
        tile_dict = self.afss_wd_stig_corr[tile_key]
        first_slice_nr = 's' + str(next(iter(tile_dict))).zfill(
            utils.SLICE_DIGITS)  # str of the first slice number of AFSS series
        g, t = tile_key.split('.')
        tile_key_full = ('g' + str(g).zfill(utils.GRID_DIGITS) + '_' + 't' + str(t).zfill(utils.TILE_DIGITS))
        plot_name = "_".join([first_slice_nr, tile_key_full, self.afss_mode, interpolation])
        plot_fn = os.path.join(self.cfg['acq']['base_dir'], 'meta', 'stats', plot_name + '.png')
        return plot_fn

    def plot_afss_series(self,
                         x_vals: np.ndarray, y_vals: np.ndarray,
                         x_fit: np.ndarray, y_fit: np.ndarray,
                         x_opt: float, y_opt: float,
                         x_orig: float, err: float,
                         path: str
                         ):
        if self.afss_mode == 'focus':  # rescale x axis to millimetres
            x_vals *= 10 ** 3
            x_fit *= 10 ** 3
            x_opt *= 10 ** 3
            x_orig *= 10 ** 3
            round_digits = 6
            unit = 'mm'
        else:
            round_digits = 3
            unit = '%'

        fig, ax = plt.subplots()
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams.update({'font.size': 12})
        ax.plot(x_vals, y_vals, 'o', label='Data')
        ax.plot(x_fit, y_fit, '-', label=f'Interpolation {self.afss_interpolation_method}, RMSE = {np.round(err, 4)}')
        ax.axvline(x_orig, color='k', linestyle=':', label=f'Previous setting: {round(x_orig, round_digits)} {unit}')
        ax.plot(x_opt, y_opt, 'o', label=f'New optimum at: {round(x_opt, round_digits)} {unit}, '
                                         f'diff = {round(x_opt - x_orig, round_digits)} {unit}')
        ax.legend()
        ax.set_title(str.split(os.path.basename(path), '.')[0] + '_series')
        x_labels = {'focus': 'Working distance [mm]', 'stig_x': 'StigX [%]', 'stig_y': 'StigY [%]'}
        plt.xlabel([val for key, val in x_labels.items() if key == self.afss_mode][0])
        plt.ylabel('Sharpness [arb.u]')
        plt.savefig(path, dpi=100)
        plt.cla()
        plt.close(fig)

    def get_average_afss_correction(self, do_filtering: bool, do_weighted_average: bool) -> Tuple[float, int]:
        #  Function for mode='Average' in f(apply_afss_corrections)
        diffs_dict = {}
        nr_of_outliers = 0
        m = self.afss_mode
        dd = {'focus': (0, 0), 'stig_x': (1, 0), 'stig_y': (1, 1)}
        for tile_key, vals in self.afss_wd_stig_corr_optima.items():
            # diff = optimum - original WD/StigX/StigY
            diffs_dict[tile_key] = vals[0] - self.afss_wd_stig_orig[tile_key][dd[m][0]][dd[m][1]]
        diffs = list(diffs_dict.values())
        if do_filtering:
            diffs_filtered = utils.filter_outliers(np.asarray(diffs))
            nr_of_outliers = len(diffs) - len(diffs_filtered)
            diffs = diffs_filtered
            for k, v in list(diffs_dict.items()):  # Remove filtered entries from helper dict (used in weights)
                if v not in diffs:
                    del (diffs_dict[k])
        avg = np.mean(diffs)
        if do_weighted_average and len(diffs) > 1:
            rmse_ = [self.afss_wd_stig_corr_optima[key][1] for key in diffs_dict.keys()]
            weights = utils.get_weights(rmse_, smallest_weight=0.3)
            if np.sum(weights) == 0:  # Prevent division by zero if by any change the sum of weight is zero
                weights[0] -= 1e-9
            avg = np.average(diffs, weights=weights)  # TODO: limit into cfg
        self.afss_avg_corr = avg
        return avg, nr_of_outliers

    def apply_afss_corrections(self) -> Tuple[float, dict, int]:
        """Apply individual tile corrections."""
        # mode = 'tile_specific'  # compute and apply corrections specific to each tile
        # mode = 'Average'  # compute average correction from results of all ref.tiles
        diffs, msgs = {}, {}
        mean_diff = 0.0
        nr_of_outs: int = 0
        mode = self.afss_mode
        consensus_modes = ['Average', 'tile_specific', 'focus_specific_stig_average']
        avg_mode = consensus_modes[self.afss_consensus_mode]
        
        if self.afss_consensus_mode == 0 or (self.afss_consensus_mode == 2 and self.afss_mode != 'focus'):
            mean_diff, nr_of_outs = self.get_average_afss_correction(
                                            do_filtering=self.afss_filter_outliers,
                                            do_weighted_average=self.afss_weighted_averaging)

        for tile_key in self.afss_wd_stig_orig:
            g, t = map(int, str.split(tile_key, '.'))
            if mode == 'focus':
                wd_orig = self.afss_wd_stig_orig[tile_key][0][0]
                if avg_mode == 'Average':
                    wd_new = mean_diff + wd_orig
                    self.gm[g][t].wd = wd_new
                    # if fit was not successful, wd difference is determined from applied average delta_wd
                    if tile_key not in self.afss_wd_stig_corr_optima:
                        diffs[tile_key] = mean_diff
                    else:
                        wd_opt = self.afss_wd_stig_corr_optima[tile_key][0]
                        diffs[tile_key] = wd_opt - wd_orig  # for logging purposes
                    msgs[tile_key] = f'AFSS: Tile {tile_key}, delta WD = {round((diffs[tile_key]) * 10 ** 6, 3)} um.'
                elif avg_mode == 'tile_specific' or avg_mode == 'focus_specific_stig_average':
                    if tile_key not in self.afss_wd_stig_corr_optima.keys():
                        wd_new = self.afss_wd_stig_orig[tile_key][0][0]
                        diffs[tile_key] = 0
                        msgs[tile_key] = f'AFSS: Tile {tile_key}, fit not reliable. Original WD will be applied.'
                    else:
                        wd_new = self.afss_wd_stig_corr_optima[tile_key][0]
                        diffs[tile_key] = wd_new - wd_orig
                        msgs[tile_key] = f'AFSS: Tile {tile_key}, delta WD = {round((diffs[tile_key]) * 10 ** 6, 3)} um.'
                    self.gm[g][t].wd = wd_new
                # Update original values by new results
                self.afss_wd_stig_orig[tile_key][0][0] = self.gm[g][t].wd
            elif mode == 'stig_x':
                stig_x_orig, stig_y_orig = self.afss_wd_stig_orig[tile_key][1]
                if avg_mode == 'Average' or avg_mode == 'focus_specific_stig_average':
                    stig_x_new = mean_diff + stig_x_orig
                    self.gm[g][t].stig_xy = [stig_x_new, stig_y_orig]
                    if tile_key not in self.afss_wd_stig_corr_optima:
                        diffs[tile_key] = mean_diff
                    else:
                        stig_x_opt = self.afss_wd_stig_corr_optima[tile_key][0]
                        diffs[tile_key] = stig_x_opt - stig_x_orig
                    msgs[tile_key] = f'AFSS: Tile {tile_key}, delta StigX = {round(diffs[tile_key], 3)} %.'
                elif avg_mode == 'tile_specific':
                    if tile_key not in self.afss_wd_stig_corr_optima:
                        self.gm[g][t].stig_xy = [stig_x_orig, stig_y_orig]
                        diffs[tile_key] = 0
                        msgs[tile_key] = f'AFSS: Tile {tile_key}, fit not reliable. Original StigX will be applied.'
                    else:
                        stig_x_new = self.afss_wd_stig_corr_optima[tile_key][0]
                        self.gm[g][t].stig_xy = [stig_x_new, stig_y_orig]
                        diffs[tile_key] = stig_x_new - stig_x_orig
                    msgs[tile_key] = f'AFSS: Tile {tile_key}, delta StigX = {round(diffs[tile_key], 3)} %.'
                # Update original values by new results
                self.afss_wd_stig_orig[tile_key][1] = self.gm[g][t].stig_xy
            elif mode == 'stig_y':
                stig_x_orig, stig_y_orig = self.afss_wd_stig_orig[tile_key][1]
                if avg_mode == 'Average' or avg_mode =='focus_specific_stig_average':
                    stig_y_new = mean_diff + stig_y_orig
                    self.gm[g][t].stig_xy = [stig_x_orig, stig_y_new]
                    if tile_key not in self.afss_wd_stig_corr_optima:
                        diffs[tile_key] = mean_diff
                    else:
                        stig_y_opt = self.afss_wd_stig_corr_optima[tile_key][0]
                        diffs[tile_key] = stig_y_opt - stig_y_orig
                    msgs[tile_key] = f'AFSS: Tile {tile_key}, delta StigY = {round(diffs[tile_key], 3)} %.'
                elif avg_mode == 'tile_specific':
                    if tile_key not in self.afss_wd_stig_corr_optima:
                        self.gm[g][t].stig_xy = [stig_x_orig, stig_y_orig]
                        diffs[tile_key] = 0
                        msgs[tile_key] = f'AFSS: Tile {tile_key}, fit not reliable. Original StigY will be applied.'
                    else:
                        stig_y_new = self.afss_wd_stig_corr_optima[tile_key][0]
                        self.gm[g][t].stig_xy[1] = stig_y_new
                        diffs[tile_key] = stig_y_new - stig_y_orig
                    msgs[tile_key] = f'AFSS: Tile {tile_key}, delta StigY = {round(diffs[tile_key], 3)} %.'
                # Update original values by new results
                self.afss_wd_stig_orig[tile_key][1] = self.gm[g][t].stig_xy
        return mean_diff, msgs, nr_of_outs

    def next_afss_mode(self):
        if not self.afss_autostig_active:
            self.afss_mode = 'focus'
        elif self.afss_mode == 'focus':
            self.afss_mode = 'stig_x'
        elif self.afss_mode == 'stig_x':
            self.afss_mode = 'stig_y'
        elif self.afss_mode == 'stig_y':
            self.afss_mode = 'focus'
        else:
            utils.log_info('Warning: undetected AFSS mode. Next run will be of type: Focus')
            self.afss_mode = 'focus'

    def get_afss_factors(self, tile_keys: dict, shuffle: bool, hyper_shuffle: bool):
        #  get list of WD or Stig perturbations to be used in automated focus/stig series
        series = np.linspace(-1, 1, self.afss_rounds)
        if shuffle:
            random.shuffle(series)
        if not hyper_shuffle:
            for key in tile_keys:
                self.afss_perturbation_series[key] = series
        else:
            fcts = np.tile(series, (len(tile_keys), 1))
            for line in fcts:
                np.random.shuffle(line)
            for i, key in enumerate(tile_keys):
                self.afss_perturbation_series[key] = fcts[i, :]
        # print(self.afss_perturbation_series)

    def reset_afss_corrections(self):
        self.afss_wd_stig_corr = {}
        self.afss_wd_stig_corr_optima = {}
        self.afss_avg_corr = None

    def afss_set_orig_wd_stig(self):
        # TODO check what if there are multiple grids with ref tiles (possibly also if inactivated grids)
        self.afss_current_round = 0
        for tile_key in self.afss_wd_stig_orig:
            grid_index, tile_index = map(int, str.split(tile_key, '.'))
            self.gm[grid_index][tile_index].wd = self.afss_wd_stig_orig[tile_key][0][0]
            self.gm[grid_index][tile_index].stig_xy = self.afss_wd_stig_orig[tile_key][1]

    # ================ EOF methods for Automated focus/stig series method ==================

    def approximate_wd_stig_in_grid(self, grid_index):
        """Approximate the working distance and stigmation parameters for all
        non-selected active tiles in the specified grid. Simple approach for
        now: use the settings of the nearest (selected) neighbour.
        TODO: Best fit of parameters using available reference tiles
        """
        active_tiles = self.gm[grid_index].active_tiles
        autofocus_ref_tiles = self.gm[grid_index].autofocus_ref_tiles()
        if active_tiles and autofocus_ref_tiles:
            for tile in active_tiles:
                min_dist = 10 ** 6
                nearest_tile = None
                for af_tile in autofocus_ref_tiles:
                    dist = self.gm[grid_index].distance_between_tiles(
                        tile, af_tile)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_tile = af_tile
                # Set focus parameters for current tile to those of the nearest
                # autofocus tile
                self.gm[grid_index][tile].wd = (
                    self.gm[grid_index][nearest_tile].wd)
                self.gm[grid_index][tile].stig_xy = (
                    self.gm[grid_index][nearest_tile].stig_xy)

    def wd_stig_diff_below_max(self, prev_wd, prev_sx, prev_sy):
        diff_wd = abs(self.sem.get_wd() - prev_wd)
        diff_sx = abs(self.sem.get_stig_x() - prev_sx)
        diff_sy = abs(self.sem.get_stig_y() - prev_sy)
        is_below = (diff_wd <= self.max_wd_diff and diff_sx <= self.max_stig_x_diff
                    and diff_sy <= self.max_stig_y_diff)
        return is_below

    def current_slice_active(self, slice_counter):
        autofocus_active, autostig_active = False, False
        if slice_counter > 0:
            autofocus_active = (slice_counter % self.interval == 0)
        if -1 < self.autostig_delay < slice_counter:
            autostig_active = ((
                                       (slice_counter - self.autostig_delay) % self.interval) == 0)
        return autofocus_active, autostig_active

    def run_zeiss_af(self, autofocus=True, autostig=True):
        """Call the SmartSEM autofocus and autostigmation routines
        separately, or the combined routine, or no routine at all. Return a
        message that the routine was completed or an error message if not.
        """
        assert autostig or autofocus
        msg = 'SmartSEM AF did not run.'
        if autofocus or autostig:
            # Switch to autofocus settings
            # TODO: allow different dwell times
            self.sem.apply_frame_settings(0, self.pixel_size, 0.8)
            sleep(0.5)
            if autofocus and autostig:
                if self.magc_mode:
                    # Run SmartSEM autofocus-autostig-autofocus sequence
                    msg = 'SmartSEM autofocus-autostig-autofocus (MagC)'
                    success = self.sem.run_autofocus()
                    sleep(0.5)
                    if success:
                        success = self.sem.run_autostig()
                        sleep(0.5)
                        if success:
                            success = self.sem.run_autofocus()
                else:
                    msg = 'SmartSEM autofocus + autostig procedure'
                    # Perform combined autofocus + autostig
                    success = self.sem.run_autofocus_stig()
            elif autofocus:
                msg = 'SmartSEM autofocus procedure'
                # Call only SmartSEM autofocus routine
                success = self.sem.run_autofocus()
            else:
                msg = 'SmartSEM autostig procedure'
                # Call only SmartSEM autostig routine
                success = self.sem.run_autostig()
        if success:
            msg = 'Completed ' + msg + '.'
        else:
            msg = 'ERROR during ' + msg + '.'
        return msg

    def run_mapfost_af(self, aberr_mode_bools=[1, 1, 1], large_aberrations=0, pixel_size=None,
                       max_wd_stigx_stigy=None) -> str:
        """
        MAPFoSt (cf. Binding et al. 2013)
        implementation by Rangoli Saxena, 2020.
        Returns:

        """
        try:
            if pixel_size is None:
                pixel_size = self.pixel_size
            self.sem.apply_frame_settings(self.MAPFOST_FRAME_RESOLUTION, pixel_size, self.mapfost_dwell_time)
            mapfost_params = {'num_aperture': self.mapfost_probe_conv,
                              'stig_rot_deg': self.mapfost_stig_rot,
                              'stig_scale': self.mapfost_stig_scale,
                              'crop_size': self.MAPFOST_PATCH_SIZE}
            corrections = autofocus_mapfost.run(self.sem.sem_api, working_distance_perturbations=[self.mapfost_wd_pert],
                                                mapfost_params=mapfost_params, max_iters=self.mapfost_max_iters,
                                                convergence_threshold=self.mapfost_conv_thresh,
                                                aberr_mode_bools=aberr_mode_bools, large_aberrations=large_aberrations,
                                                max_wd_stigx_stigy=max_wd_stigx_stigy)
            msg = 'Completed MAPFoSt AF. \n List of corrections : \n' + str(corrections)
        except Exception as e:
            msg = f'CTRL: Exception ({str(e)}) during MAPFoSt AF.'
        return msg

    def calibrate_mapfost_af(self, calib_mode) -> str:
        """
        MAPFoSt calibration
        Rangoli Saxena, 2020.
        Still in development. In case of issues, please raise them on github to help make this better.
        Returns: mapfost calibration parameters

        """
        try:
            self.sem.apply_frame_settings(self.MAPFOST_FRAME_RESOLUTION, self.pixel_size, self.mapfost_dwell_time)
            mapfost_params = {'num_aperture': self.mapfost_probe_conv,
                              'stig_rot_deg': 0,
                              'stig_scale': [1., 1.],
                              'crop_size': self.MAPFOST_PATCH_SIZE}
            calib_param = autofocus_mapfost.calibrate(self.sem.sem_api, mapfost_params=mapfost_params,
                                                      calib_mode=calib_mode)
            msg = calib_param
        except Exception as e:
            msg = f'CTRL: Exception ({str(e)}) during MAPFoSt AF.'
        return msg

    # ================ Below: methods for heuristic autofocus ==================

    def prepare_tile_for_heuristic_af(self, tile_img, tile_key):
        """Crop tile_img provided as numpy array. Save in dictionary with
        tile_key.
        """
        height, width = tile_img.shape[0], tile_img.shape[1]
        # Crop image to 512 x 512 central area
        self.img[tile_key] = tile_img[int(height / 2 - 256):int(height / 2 + 256),
                             int(width / 2 - 256):int(width / 2 + 256)]

    def process_image_for_heuristic_af(self, tile_key):
        """Compute single-image estimators as described in Appendix A of
        Binding et al. (2013).
        """

        # The image from the dictionary self.img is provided as a numpy array
        # and already cropped to 512 x 512 pixels
        img = self.img[tile_key]
        mean = int(np.mean(img))
        # Recast as int16 and subtract mean
        img = img.astype(np.int16)
        img -= mean
        # Autocorrelation
        norm = np.sum(img ** 2)
        autocorr = fftconvolve(img, img[::-1, ::-1]) / norm
        height, width = autocorr.shape[0], autocorr.shape[1]
        # Crop to 64 x 64 px central region
        autocorr = autocorr[int(height / 2 - 32):int(height / 2 + 32),
                   int(width / 2 - 32):int(width / 2 + 32)]
        # Calculate coefficients
        fi = self.muliply_with_mask(autocorr, self.fi_mask)
        fo = self.muliply_with_mask(autocorr, self.fo_mask)
        apx = self.muliply_with_mask(autocorr, self.apx_mask)
        amx = self.muliply_with_mask(autocorr, self.amx_mask)
        apy = self.muliply_with_mask(autocorr, self.apy_mask)
        amy = self.muliply_with_mask(autocorr, self.amy_mask)
        # Check if tile_key not in dictionary yet
        if not (tile_key in self.foc_est):
            self.foc_est[tile_key] = []
        if not (tile_key in self.astgx_est):
            self.astgx_est[tile_key] = []
        if not (tile_key in self.astgy_est):
            self.astgy_est[tile_key] = []
        # Calculate single-image estimators for current tile key
        if len(self.foc_est[tile_key]) > 1:
            self.foc_est[tile_key].pop(0)
        self.foc_est[tile_key].append(
            (fi - fo) / (fi + fo))
        if len(self.astgx_est[tile_key]) > 1:
            self.astgx_est[tile_key].pop(0)
        self.astgx_est[tile_key].append(
            (apx - amx) / (apx + amx))
        if len(self.astgy_est[tile_key]) > 1:
            self.astgy_est[tile_key].pop(0)
        self.astgy_est[tile_key].append(
            (apy - amy) / (apy + amy))

    def get_heuristic_corrections(self, tile_key):
        """Use the estimators to calculate corrections."""

        if (len(self.foc_est[tile_key]) > 1
                and len(self.astgx_est[tile_key]) > 1
                and len(self.astgy_est[tile_key]) > 1):

            wd_corr = (self.heuristic_calibration[0]
                       * (self.foc_est[tile_key][0] - self.foc_est[tile_key][1]))
            a1 = (self.heuristic_calibration[1]
                  * (self.astgx_est[tile_key][0] - self.astgx_est[tile_key][1]))
            a2 = (self.heuristic_calibration[2]
                  * (self.astgy_est[tile_key][0] - self.astgy_est[tile_key][1]))

            ax_corr = a1 * cos(self.rot_angle) - a2 * sin(self.rot_angle)
            ay_corr = a1 * sin(self.rot_angle) + a2 * cos(self.rot_angle)
            ax_corr *= self.scale_factor
            ay_corr *= self.scale_factor
            # Check if results are within permissible range
            within_range = (abs(wd_corr / 1000) <= self.max_wd_diff
                            and abs(ax_corr) <= self.max_stig_x_diff
                            and abs(ay_corr) <= self.max_stig_y_diff)
            if within_range:
                # Store corrections for this tile in dictionary
                self.wd_stig_corr[tile_key] = [wd_corr / 1000, ax_corr, ay_corr]
            else:
                self.wd_stig_corr[tile_key] = [0, 0, 0]

            return wd_corr, ax_corr, ay_corr, within_range
        else:
            return None, None, None, False

    def get_heuristic_average_grid_correction(self, grid_index):
        """Use the available corrections for the reference tiles in the grid
        specified by grid_index to calculate the average corrections for the
        entire grid.
        """
        wd_corr = []
        stig_x_corr = []
        stig_y_corr = []
        for tile_key in self.wd_stig_corr:
            g = int(tile_key.split('.')[0])
            if g == grid_index:
                wd_corr.append(self.wd_stig_corr[tile_key][0])
                stig_x_corr.append(self.wd_stig_corr[tile_key][1])
                stig_y_corr.append(self.wd_stig_corr[tile_key][2])
        if wd_corr:
            return (mean(wd_corr), mean(stig_x_corr), mean(stig_y_corr))
        else:
            return (None, None, None)

    def apply_heuristic_tile_corrections(self):
        """Apply individual tile corrections."""
        for tile_key in self.wd_stig_corr:
            g, t = tile_key.split('.')
            g, t = int(g), int(t)
            self.gm[g][t].wd += self.wd_stig_corr[tile_key][0]
            self.gm[g][t].stig_xy[0] += self.wd_stig_corr[tile_key][1]
            self.gm[g][t].stig_xy[1] += self.wd_stig_corr[tile_key][2]

    def make_heuristic_weight_function_masks(self):
        # Parameters as given in Appendix A of Binding et al. 2013
        α = 6
        β = 0.5
        γ = 3
        δ = 0.5
        ε = 9

        for i in range(self.ACORR_WIDTH):
            for j in range(self.ACORR_WIDTH):
                x, y = i - self.ACORR_WIDTH / 2, j - self.ACORR_WIDTH / 2
                r = sqrt(x ** 2 + y ** 2)
                if r == 0:
                    r = 1  # Prevent division by zero
                sinφ = x / r
                cosφ = y / r
                exp_astig = exp(-r ** 2 / α) - exp(-r ** 2 / β)

                # Six masks for the calculation of coefficients:
                # fi, fo, apx, amx, apy, amy
                self.fi_mask[i, j] = exp(-r ** 2 / γ) - exp(-r ** 2 / δ)
                self.fo_mask[i, j] = exp(-r ** 2 / ε) - exp(-r ** 2 / γ)
                self.apx_mask[i, j] = sinφ ** 2 * exp_astig
                self.amx_mask[i, j] = cosφ ** 2 * exp_astig
                self.apy_mask[i, j] = 0.5 * (sinφ + cosφ) ** 2 * exp_astig
                self.amy_mask[i, j] = 0.5 * (sinφ - cosφ) ** 2 * exp_astig

    def muliply_with_mask(self, autocorr, mask):
        numerator_sum = 0
        norm = 0
        for i in range(self.ACORR_WIDTH):
            for j in range(self.ACORR_WIDTH):
                numerator_sum += autocorr[i, j] * mask[i, j]
                norm += mask[i, j]
        return numerator_sum / norm

    def reset_heuristic_corrections(self):
        self.foc_est = {}
        self.astgx_est = {}
        self.astgy_est = {}
        self.wd_stig_corr = {}
