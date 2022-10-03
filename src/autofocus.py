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

import json
import os.path
from typing import Union, Tuple, Optional, List, Any

import numpy as np
from math import sqrt, exp, sin, cos
from statistics import mean

import skimage.io

import utils
from time import sleep, time
from scipy.signal import correlate2d, fftconvolve
import autofocus_mapfost
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
# plt.rcParams['figure.figsize'] = (8, 6)
# plt.rcParams.update({'font.size': 16})

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
        self.afss_current_round = 0  # position of current WD/stig deviation within AFSS series
        self.afss_offset = json.loads(self.cfg['autofocus']['afss_offset'])  # skip N slices before first AFSS activation
        self.afss_next_activation = 0  # slice nr of nearest planned AFSS run
        self.afss_perturbation_series = []  # series that holds factors by which is the wd/stig delta multiplied
        self.afss_wd_stig_orig = {}  # original values before the AFSS started: dict = {tile_keys: [wd, (sx,sy)]}
        self.afss_wd_stig_corr = {}  #  dict = {tile_keys: {slice_nrs: [wd, (sx,sy), sharpness, img_full_path]}}
        self.afss_wd_stig_corr_optima = {}  # Computed corrections AFSS: dict = {tile_keys: wd/stig opt.val}
        self.afss_mode = 'stig_y'  # 'focus' 'stig_x' 'stig_y'
        self.afss_consensus_mode = int(self.cfg['autofocus']['afss_consensus_mode'])  # 0: 'Average' or 1: 'Tile specific'
        self.afss_drift_corrected = (self.cfg['autofocus']['afss_drift_corrected'].lower() == 'true')
        self.afss_active = False

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

    # ================ Below: methods for Automated focus/stig series method ==================
    def process_afss_collections(self):
        for tile_key in self.afss_wd_stig_corr:
            print(f'processing collection: {tile_key} ')
            filenames = []
            basenames = []
            for slice_nr in self.afss_wd_stig_corr[tile_key]:
                img_path = self.afss_wd_stig_corr[tile_key][slice_nr][3]
                filenames.append(img_path)
                basenames.append(os.path.basename(img_path))
            print(f'collection filenames: {filenames} \n')
            ic = utils.load_image_collection(filenames)
            ic_reg, cumm_shifts = utils.register_image_collection(ic)
            ic_cr = utils.crop_image_collection(ic_reg, cumm_shifts)
            coll_sharpness = utils.get_collection_sharpness(ic_cr, metric='contrast')  # based on custom mask defined
            # by shape of images in the collection
            # Fill the results' dict with sharpness values from drift-corrected image collection
            for i, slice_nr in enumerate(self.afss_wd_stig_corr[tile_key]):
                print(f'Populating {tile_key}, slice_nr: {slice_nr} with sharpness value: {coll_sharpness[i]}\n')
                self.afss_wd_stig_corr[tile_key][slice_nr][2] = coll_sharpness[i]
                reg_img_path = os.path.join(self.cfg['acq']['base_dir'], 'meta', 'stats', basenames[i])
                skimage.io.imsave(reg_img_path, ic_cr[i])
        print(f'processing collections finished...')

    def fit_afss_collections(self, plot_results=True):
        mode = self.afss_mode
        # print(self.afss_wd_stig_corr)
        for tile_key in self.afss_wd_stig_corr:
            # print(f'fitting collection: {tile_key}')
            x_vals, y_vals = [], []
            opt = 0.1
            cfs = []
            tile_dict = self.afss_wd_stig_corr[tile_key]  # values of particular tile to be processed

            # read the values (wd/stig_x/stig_y, sharpness)
            if mode == 'focus':
                for slice_nr in tile_dict:
                    x_vals.append(tile_dict[slice_nr][0])  # WD series
            elif mode == 'stig_x':
                for slice_nr in tile_dict:
                    x_vals.append(tile_dict[slice_nr][1][0])  # StigX deviation
            elif mode == 'stig_y':
                for slice_nr in tile_dict:
                    x_vals.append(tile_dict[slice_nr][1][1])  # StigX deviation
            for slice_nr in tile_dict:
                y_vals.append(tile_dict[slice_nr][2])  # list of sharpness values

            # SPLINE INTERPOLATION
            f1 = interp1d(x_vals, y_vals, kind='cubic')
            x_fit = np.linspace(min(x_vals), max(x_vals), num=101, endpoint=True)
            y_fit = f1(x_fit)
            x_opt, y_opt = x_fit[np.argmax(y_fit)], max(y_fit)  # x,y coordinates of optimal value
            self.afss_wd_stig_corr_optima[tile_key] = x_opt

            # # POLYNOMIAL FIT
            # wd1, wd2 = 0, 5
            # WDs = np.linspace(wd1, wd2, len(filenames))
            # cfs = np.polyfit(WDs, coll_sharpness_edg, 2)
            # x = np.linspace(wd1, wd2, 1000)
            # y = cfs[0] * x ** 2 + cfs[1] * x + cfs[2]
            # opt = -cfs[1] / (2 * cfs[0])


            # Save resulting plot into stats folder
            if plot_results:
                plot_path = self.generate_afss_plot_path(tile_key)
                self.plot_afss_series(x_vals=np.asarray(x_vals), y_vals=np.asarray(y_vals),
                                      x_fit=x_fit, y_fit=y_fit,
                                      x_opt=x_opt, y_opt=y_opt,
                                      path=plot_path)

        self.afss_wd_stig_corr = {} # Reset the correction dictionary to prepare it for next afss run

    def generate_afss_plot_path(self, tile_key: str) -> str:
        # Generate plot name: basedir + 'slice_nr'_'grid_nr'_'tile_nr'_'focus/x_stig/y_stig_fit'.png
        tile_dict = self.afss_wd_stig_corr[tile_key]
        first_slice_nr = 's' + str(next(iter(tile_dict))).zfill(
            utils.SLICE_DIGITS)  # str of the first slice number of AFSS series
        g, t = tile_key.split('.')
        tile_key_full = ('g' + str(g).zfill(utils.GRID_DIGITS) + '_' + 't' + str(t).zfill(utils.TILE_DIGITS))
        plot_name = "_".join([first_slice_nr, tile_key_full, self.afss_mode])
        plot_fn = os.path.join(self.cfg['acq']['base_dir'], 'meta', 'stats', plot_name + '.png')
        return plot_fn

    def plot_afss_series(self,
                         x_vals: np.ndarray, y_vals: np.ndarray,
                         x_fit: np.ndarray, y_fit: np.ndarray,
                         x_opt: float, y_opt: float,
                         path: str
                         ):
        # y = cfs[0] * x ** 2 + cfs[1] * x + cfs[2]
        fig, ax = plt.subplots()
        if self.afss_mode == 'focus':   # rescale x axis to millimetres
            x_vals *= 10**3
            x_fit *=10**3
            x_opt *=10**3
            round_digits = 6
            unit = 'mm'
        else:
            round_digits = 3
            unit = '%'
        ax.plot(x_vals, y_vals, 'o', label='Data')
        ax.plot(x_fit, y_fit, '-', label='Spline interp.')
        ax.plot(x_opt, y_opt, 'o', label=f'Optimum at: {round(x_opt, round_digits)} {unit}')
        ax.legend()
        ax.set_title(str.split(os.path.basename(path), '.')[0] + '_series')
        xlabels = {'focus': 'Working distance [mm]', 'stig_x': 'StigX [%]', 'stig_y': 'StigY [%]'}
        plt.xlabel([val for key, val in xlabels.items() if key==self.afss_mode][0])
        plt.ylabel('Sharpness [arb.u]')
        plt.savefig(path, dpi=100)

    def next_afss_mode(self):
        if self.autostig_delay == -1:
            self.afss_mode = 'focus'
        elif self.afss_mode == 'focus':
            self.afss_mode = 'stig_x'
        elif self.afss_mode == 'stig_x':
            self.afss_mode = 'stig_y'
        elif self.afss_mode == 'stig_y':
            self.afss_mode = 'focus'
        else:
            utils.log_info('Warning: undetected AFSS mode. Next run will be of type: Focus.')
            self.afss_mode = 'focus'

    def get_afss_factors(self):
        #  get list of WD or Stig perturbations to be used in automated focus/stig series
        self.afss_perturbation_series = np.linspace(-1, 1, self.afss_rounds)

    def afss_new_vals_verified(self) -> bool:
        mode = self.afss_mode
        # Check that all new WDs and Stigmator values of ref. tiles are below WD/Stig thresholds
        is_below = True
        diff_wd, diff_sx, diff_sy = 0, 0, 0
        for tile_key, value in self.afss_wd_stig_corr_optima.items():
            if mode == 'focus':
                diff_wd = abs(value - self.afss_wd_stig_orig[tile_key][0])
            elif mode == 'stig_x':
                diff_sx = abs(value - self.afss_wd_stig_orig[tile_key][1][0])
            elif mode == 'stig_y':
                diff_sy = abs(value - self.afss_wd_stig_orig[tile_key][1][1])
            is_below &= (diff_wd <= self.max_wd_diff and diff_sx <= self.max_stig_x_diff
                         and diff_sy <= self.max_stig_y_diff)
        return is_below

    def get_average_afss_correction(self):
        mode = self.afss_mode
        # Function for mode='Average' in f(apply_afss_corrections)
        diffs = []
        for tile_key in self.afss_wd_stig_corr_optima:
            opt = self.afss_wd_stig_corr_optima[tile_key]
            if mode == 'focus':
                #
                diffs.append(opt - self.afss_wd_stig_orig[tile_key][0])
            elif mode == 'stig_x':
                diffs.append(opt - self.afss_wd_stig_orig[tile_key][1][0])
            elif mode == 'stig_y':
                diffs.append(opt - self.afss_wd_stig_orig[tile_key][1][1])
        return np.mean(diffs)

    def apply_afss_corrections(self) -> Tuple[float, dict, dict]:
        # utils.log_info('AFSS', 'Applying corrections to WD/STIG:')
        # TODO refactor
        consensus_modes = ['Average', 'tile_specific']
        avg_mode = consensus_modes[self.afss_consensus_mode]
        mode = self.afss_mode
        """Apply individual tile corrections."""
        # mode = 'tile_specific'  # compute and apply corrections specific to each tile
        # mode = 'Average'  # compute average correction from results of all ref.tiles
        diffs = {}
        msgs = {}
        mean_diff = None
        if avg_mode == 'Average':
            mean_diff = self.get_average_afss_correction()

        for tile_key in self.afss_wd_stig_corr_optima:
            g, t = map(int, str.split(tile_key, '.'))
            if mode == 'focus':
                wd_orig = self.afss_wd_stig_orig[tile_key][0]
                if avg_mode == 'Average':
                    self.gm[g][t].wd = mean_diff + wd_orig
                    wd_new = self.afss_wd_stig_corr_optima[tile_key]
                    diffs[tile_key] = wd_new - wd_orig  # for logging purposes
                    msgs[tile_key] = f'AFSS: Tile {tile_key}, delta WD = {round((wd_new - wd_orig) * 10 ** 6, 3)} um.'
                elif avg_mode == 'tile_specific':
                    wd_new = self.afss_wd_stig_corr_optima[tile_key]
                    diffs[tile_key] = wd_new - wd_orig  # for logging purposes
                    self.gm[g][t].wd = wd_new
                    msgs[tile_key] = f'AFSS: Tile {tile_key}, delta WD = {round((wd_new - wd_orig)*10**6, 3)} um.'
                    # utils.log_info(msgs[tile_key])
                else:
                    #
                    utils.log_info('AFSS:', 'Wrong mode in apply_afss_corrections !')
                # Update original values by new results
                self.afss_wd_stig_orig[tile_key][0] = self.gm[g][t].wd
            elif mode == 'stig_x':
                stig_x_orig, stig_y_orig = self.afss_wd_stig_orig[tile_key][1]
                if avg_mode == 'Average':
                    self.gm[g][t].stig_xy = [stig_x_orig + mean_diff, stig_y_orig]
                    stig_x_new = self.afss_wd_stig_corr_optima[tile_key]
                    diffs[tile_key] = stig_x_new - stig_x_orig
                    msgs[tile_key] = f'AFSS: Tile {tile_key}, delta StigX = {round(stig_x_new - stig_x_orig, 3)} %.'
                elif avg_mode == 'tile_specific':
                    stig_x_new = self.afss_wd_stig_corr_optima[tile_key]
                    self.gm[g][t].stig_xy = [stig_x_new, stig_y_orig]
                    diffs[tile_key] = stig_x_new - stig_x_orig
                    msgs[tile_key] = f'AFSS: Tile {tile_key}, delta StigX = {round(stig_x_new - stig_x_orig, 3)} %.'
                # Update original values by new results
                self.afss_wd_stig_orig[tile_key][1] = self.gm[g][t].stig_xy
            elif mode == 'stig_y':
                stig_x_orig, stig_y_orig = self.afss_wd_stig_orig[tile_key][1]
                if avg_mode == 'Average':
                    self.gm[g][t].stig_xy = [stig_x_orig, stig_y_orig + mean_diff]
                    stig_y_new = self.afss_wd_stig_corr_optima[tile_key]
                    diffs[tile_key] = stig_y_new - stig_y_orig
                    msgs[tile_key] = f'AFSS: Tile {tile_key}, delta StigY = {round(stig_y_new - stig_y_orig, 3)} %.'
                elif avg_mode == 'tile_specific':
                    stig_y_new = self.afss_wd_stig_corr_optima[tile_key]
                    diffs[tile_key] = stig_y_new - stig_y_orig
                    self.gm[g][t].stig_xy = [stig_x_orig, stig_y_new]
                    msgs[tile_key] = f'AFSS: Tile {tile_key}, delta StigY = {round(stig_y_new - stig_y_orig, 3)} %.'
                # Update original values by new results
                self.afss_wd_stig_orig[tile_key][1] = self.gm[g][t].stig_xy

        return mean_diff, diffs, msgs

    # TODO: redundant, staged for removal
    def update_afss_wd_stig_orig(self, tile_key, value):
        self.afss_wd_stig_orig[tile_key][0] = value

    def reset_afss_corrections(self):  # TODO: merge with 'reset_afss_series' ?
        self.afss_wd_stig_corr = {}
        self.afss_wd_stig_corr_optima = {}

    def reset_afss_series(self):  # TODO: more explanationatory name
        # TODO check what if there are multiple grids with ref tiles (possibly also if inactivated grids)
        self.afss_current_round = 0
        for tile_key in self.afss_wd_stig_orig:
            grid_index, tile_index = map(int, str.split(tile_key, '.'))
            self.gm[grid_index][tile_index].wd = self.afss_wd_stig_orig[tile_key][0]
            self.gm[grid_index][tile_index].stig_xy = self.afss_wd_stig_orig[tile_key][1]
        # autofocus_ref_tiles = self.gm[grid_index].autofocus_ref_tiles()
        # for tile_index in autofocus_ref_tiles:
        #     key = f'{grid_index}.{tile_index}'
        #     self.gm[grid_index][tile_index].wd = self.afss_wd_stig_orig[key][0]
        #     self.gm[grid_index][tile_index].stig_xy = self.afss_wd_stig_orig[key][1]

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
