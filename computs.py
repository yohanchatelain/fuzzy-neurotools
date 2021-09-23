#!/usr/bin/env python3

from nibabel.processing import resample_from_to
import scipy
import scipy.spatial
import nibabel as nib
import numpy as np
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import pickle as pkl
import inspect
import os
from nilearn.image import math_img
#import seaborn as sns
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from lib import bland_altman
from lib import dice
from statistics import mean


############# COMPUTES STD. DEV.

def compute_var(f1, f2, file_name, f3=None):
    # Compute std. between tool image samples
    f1 = nib.load(f1)
    f2 = nib.load(f2)
    img_concat = nib.funcs.concat_images([f1, f2], check_affines=True, axis=None)
    img_std = np.std(img_concat.get_fdata(), axis=3)
    # Extract background
    # f1_mask = set_zero_to_nan(f1)
    # f2_mask = set_zero_to_nan(f2)
    # bg_ = np.logical_and(np.isnan(f1_mask.get_fdata()), np.isnan(f2_mask.get_fdata()))

    # Compute variance between fuzzy image samples
    if f3 is not None:
        # f3 = nib.load(f3)
        # f3_mask = set_zero_to_nan(f3)
        # bg_ = np.logical_and(np.isnan(f1_mask.get_fdata()), np.isnan(f2_mask.get_fdata()), np.isnan(f3_mask.get_fdata()))
        img_concat = nib.funcs.concat_images([f1, f2, f3], check_affines=True, axis=None)
        img_var = np.var(img_concat.get_fdata(), axis=3)
        # set background voxels as nan to not display later
        # img_var[bg_] = np.nan
        nft_img = nib.Nifti1Image(img_var, f1.affine, header=f1.header)
        nib.save(nft_img, './figures/map-on-surf/std/{}-var.nii.gz'.format(file_name))
        img_std = np.std(img_concat.get_fdata(), axis=3)

    img_std[img_std == 0] = np.nan
    # set background voxels as nan to not display later
    # img_std[bg_] = np.nan
    # mean_ = np.nanmean(img_std)
    # print('mean of {} is {}'.format(file_name, mean_))

    nft_img = nib.Nifti1Image(img_std, f1.affine, header=f1.header)
    nib.save(nft_img, './figures/map-on-surf/std/{}-std.nii.gz'.format(file_name))


def combine_var(f1, f2, file_name):
    var_f1 = nib.load(f1)
    var_f2 = nib.load(f2)
    var_f2_res = resample_from_to(var_f2, var_f1, order=0)
    # to combine two image variances, we use: var(x+y) = var(x) + var(y) + 2*cov(x,y)
    # and since the correlation between two arrays are so weak, we droped `2*cov(x,y)` from the formula
    # bg_ = np.logical_and(np.isnan(var_f1.get_fdata()), np.isnan(var_f2_res.get_fdata()))
    combine_var = np.nan_to_num(var_f1.get_fdata()) + np.nan_to_num(var_f2_res.get_fdata())
    std_ = np.sqrt(combine_var)
    std_[std_ == 0] = np.nan
    #std_[bg_] = np.nan
    # mean_ = np.nanmean(std_)
    # print('mean of {} is {}'.format(file_name, mean_))

    nft_img = nib.Nifti1Image(std_, var_f1.affine, header=var_f1.header)
    nib.save(nft_img, './figures/map-on-surf/std/{}-std.nii.gz'.format(file_name))


def set_zero_to_nan(data_img):
    # Set masking using NaN's
    data_orig = data_img.get_data()

    if np.any(np.isnan(data_orig)):
        # Already using NaN
        data_img_nan = data_img
    else:
        # Replace zeros by NaNs
        data_nan = data_orig.astype(float)
        data_nan[data_nan == 0] = np.nan
        # Save as image
        data_img_nan = nib.Nifti1Image(data_nan, data_img.get_affine())

    return(data_img_nan)


def var_between_tool(tool_results):
    ## thresholded
    fsl_ = tool_results['fsl']['act_deact']
    afni_ = tool_results['afni']['act_deact']
    spm_ = tool_results['spm']['act_deact']
    # resampling first image on second image
    spm_res = resample_imgs(spm_, fsl_)
    afni_res = resample_imgs(afni_, fsl_)
    # compute variances
    compute_var(fsl_, afni_res, 'fsl-afni-thresh', f3=None)
    compute_var(fsl_, spm_res, 'fsl-spm-thresh', f3=None)
    spm_res = resample_imgs(spm_, afni_)
    compute_var(afni_, spm_res, 'afni-spm-thresh', f3=None)

    # unthresholded
    fsl_ = tool_results['fsl']['stat_file']
    afni_ = tool_results['afni']['stat_file']
    spm_ = tool_results['spm']['stat_file']
    # resampling first image on second image
    spm_res = resample_imgs(spm_, fsl_)
    afni_res = resample_imgs(afni_, fsl_)
    # compute variances
    compute_var(fsl_, afni_res, 'fsl-afni-unthresh', f3=None)
    compute_var(fsl_, spm_res, 'fsl-spm-unthresh', f3=None)
    spm_res = resample_imgs(spm_, afni_)
    compute_var(afni_, spm_res, 'afni-spm-unthresh', f3=None)


def var_between_fuzzy(mca_results):
    ## thresholded
    fsl_1 = mca_results['fsl'][1]['act_deact']
    fsl_2 = mca_results['fsl'][2]['act_deact']
    fsl_3 = mca_results['fsl'][3]['act_deact']
    afni_1 = mca_results['afni'][1]['act_deact']
    afni_2 = mca_results['afni'][2]['act_deact']
    afni_3 = mca_results['afni'][3]['act_deact']
    spm_1 = mca_results['spm'][1]['act_deact']
    spm_2 = mca_results['spm'][2]['act_deact']
    spm_3 = mca_results['spm'][3]['act_deact']
    # compute variances
    compute_var(fsl_1, fsl_2, 'fuzzy-fsl-thresh', f3=fsl_3)
    compute_var(afni_1, afni_2, 'fuzzy-afni-thresh', f3=afni_3)
    compute_var(spm_1, spm_2, 'fuzzy-spm-thresh', f3=spm_3)
    # combine variances
    var1 = './figures/map-on-surf/std/fuzzy-fsl-thresh-var.nii.gz'
    var2 = './figures/map-on-surf/std/fuzzy-afni-thresh-var.nii.gz'
    var3 = './figures/map-on-surf/std/fuzzy-spm-thresh-var.nii.gz'
    combine_var(var1, var2, 'fuzzy-fsl-afni-thresh')
    combine_var(var1, var3, 'fuzzy-fsl-spm-thresh')
    combine_var(var2, var3, 'fuzzy-afni-spm-thresh')

    # unthresholded
    fsl_1 = mca_results['fsl'][1]['stat_file']
    fsl_2 = mca_results['fsl'][2]['stat_file']
    fsl_3 = mca_results['fsl'][3]['stat_file']
    afni_1 = mca_results['afni'][1]['stat_file']
    afni_2 = mca_results['afni'][2]['stat_file']
    afni_3 = mca_results['afni'][3]['stat_file']
    spm_1 = mca_results['spm'][1]['stat_file']
    spm_2 = mca_results['spm'][2]['stat_file']
    spm_3 = mca_results['spm'][3]['stat_file']
    # compute variances
    compute_var(fsl_1, fsl_2, 'fuzzy-fsl-unthresh', f3=fsl_3)
    compute_var(afni_1, afni_2, 'fuzzy-afni-unthresh', f3=afni_3)
    compute_var(spm_1, spm_2, 'fuzzy-spm-unthresh', f3=spm_3)
    # combine variances
    var1 = './figures/map-on-surf/std/fuzzy-fsl-unthresh-var.nii.gz'
    var2 = './figures/map-on-surf/std/fuzzy-afni-unthresh-var.nii.gz'
    var3 = './figures/map-on-surf/std/fuzzy-spm-unthresh-var.nii.gz'
    combine_var(var1, var2, 'fuzzy-fsl-afni-unthresh')
    combine_var(var1, var3, 'fuzzy-fsl-spm-unthresh')
    combine_var(var2, var3, 'fuzzy-afni-spm-unthresh')


def get_ratio():
    path_ = './figures/map-on-surf/std/'
    for type_ in ['thresh', 'unthresh']:
        for pair_ in ['fsl-spm', 'fsl-afni', 'afni-spm']:
            img1_ = nib.load('{}{}-{}-std.nii.gz'.format(path_, pair_, type_))
            img2_ = nib.load('{}fuzzy-{}-{}-std.nii.gz'.format(path_, pair_, type_))
            img1_data = np.nan_to_num(img1_.get_fdata())
            img2_data = np.nan_to_num(img2_.get_fdata())
            ratio_ = img1_data / img2_data
            nft_img = nib.Nifti1Image(ratio_, img1_.affine, header=img1_.header)
            nib.save(nft_img, './figures/map-on-surf/std/ratio-{}-{}.nii.gz'.format(pair_, type_))

            # img1_data[img1_data < 0.1] = 0
            # img2_data[img2_data < 0.1] = 0
            # thresh_ratio = img1_data / img2_data
            # #ones_ratio = np.where((0.1<thresh_ratio) & (thresh_ratio< 2) , 1, np.nan)
            # thresh_ratio[thresh_ratio > 2] = 0
            # thresh_ratio[thresh_ratio < 0.1] = 0
            # nft_img = nib.Nifti1Image(thresh_ratio, img1_.affine, header=img1_.header)
            # nib.save(nft_img, './figures/map-on-surf/std/ratioT-{}-{}.nii.gz'.format(pair_, type_))


def resample_imgs(f1, f2):
    # Resample data1 on data2 using nearest neighbours
    img1 = nib.load(f1)
    img2 = nib.load(f2)
    img1_res = resample_from_to(img1, img2, order=0)
    #spm_res = nib.Nifti1Image(spm_d, fsl_n.affine, header=fsl_n.header)
    file_name = f1.replace('.nii.gz', '_res.nii.gz')
    nib.save(img1_res, file_name)
    return file_name


def cluster_std(bt_, wt_, img_f1, tool_, type_):
    Z = np.zeros(bt_.shape)
    #M = np.full(bt_.shape, -1)
    O = np.ones(bt_.shape)
    bt_ = np.nan_to_num(bt_)
    wt_ = np.nan_to_num(wt_)

    # Upper maps
    bt_class1 = np.where((bt_<0.005), bt_, np.nan)# & (wt_>3))
    bt_data1 = np.reshape(bt_class1, -1)
    wt_class1 = np.where((bt_<0.005), wt_, np.nan)# & (wt_>3))
    wt_data1 = np.reshape(wt_class1, -1)
    upper_map = np.where((bt_<0.005), O, np.nan)# & (wt_>3))
    img_d_img = nib.Nifti1Image(upper_map, img_f1.affine, header=img_f1.header)
    nib.save(img_d_img, './figures/map-on-surf/std/clusters/upper-maps-{}-{}.nii.gz'.format(type_, tool_))

    # Lower maps
    bt_class2 = np.where((wt_<0.1), bt_, np.nan)#
    bt_data2 = np.reshape(bt_class2, -1)
    wt_class2 = np.where((wt_<0.1), wt_, np.nan)#
    wt_data2 = np.reshape(wt_class2, -1)
    lower_map = np.where((wt_<0.1), O, np.nan)#
    img_d_img = nib.Nifti1Image(lower_map, img_f1.affine, header=img_f1.header)
    nib.save(img_d_img, './figures/map-on-surf/std/clusters/lower-maps-{}-{}.nii.gz'.format(type_, tool_))

    # Correlated maps
    # (0.5<bt_/wt_) & (bt_/wt_< 2) & (wt_ > 0.1) & (bt_ > 0.1)
    bt_class3 = np.where((0.5<bt_/wt_) & (bt_/wt_< 2) & (wt_ > 0.1) & (bt_ > 0.1), bt_, np.nan)#
    bt_data3 = np.reshape(bt_class3, -1)
    wt_class3 = np.where((0.5<bt_/wt_) & (bt_/wt_< 2) & (wt_ > 0.1) & (bt_ > 0.1), wt_, np.nan)#
    wt_data3 = np.reshape(wt_class3, -1)
    corr_map = np.where((0.5<bt_/wt_) & (bt_/wt_< 2) & (wt_ > 0.1) & (bt_ > 0.1), bt_/wt_, np.nan)#
    img_d_img = nib.Nifti1Image(corr_map, img_f1.affine, header=img_f1.header)
    nib.save(img_d_img, './figures/map-on-surf/std/clusters/correlated-maps-{}-{}.nii.gz'.format(type_, tool_))

    # s0, s1, s2 = corr_map.shape
    # corr_map2 = np.zeros(shape=corr_map.shape)
    # corr_map2[int(s0/2),int(s1/2),int(s2/2)] = 100
    # img_d_img2 = nib.Nifti1Image(corr_map2, img_f1.affine, header=img_f1.header)
    # nib.save(img_d_img2, './figures/map-on-surf/std/clusters/artifact.nii.gz')

    return bt_data1, wt_data1, bt_data2, wt_data2, bt_data3, wt_data3, corr_map, upper_map


def plot_pair_corr_variance(bt, wt, type_, ind_, ax, tool_=None, append_=None):
    img_f1 = nib.load(bt)
    img_f2 = nib.load(wt)
    if img_f1.shape != img_f2.shape:
        img_f2 = resample_from_to(img_f2, img_f1, order=0)

    bt_ = img_f1.get_fdata()
    wt_ = img_f2.get_fdata()
    data1 = np.reshape(bt_, -1)
    data1 = np.nan_to_num(data1)
    data2 = np.reshape(wt_, -1)
    data2 = np.nan_to_num(data2)

    slope, intercept, r, p, stderr = scipy.stats.linregress(data1, data2)
    #line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'
    y = intercept + slope * data1

    t1, t2 = tool_.split('-')
    label_ = "{} and {}".format(t1.upper(), t2.upper()) #'{}'.format(tool_.upper())
    # if single: label_ = '{}'.format(tool_.upper())

    if append_ is not None:
        ax.plot(data1, data2, linewidth=0, marker='o', color='indianred', alpha=.5, label=label_)
        ax.plot(data1, y, color='black', alpha=.5, label='Regression line')
        if append_ == 'left':
            ax.set_ylabel('Within tool variation (std.)')
        #ax.set_xlim([-0.35, 5])
        #ax.set_ylim([-0.35, 2.5])
        ax.legend(facecolor='white', loc='upper right')
        #ax.set_xscale('log')
    else:
        bt_data1, wt_data1, bt_data2, wt_data2, bt_data3, wt_data3, corr_map, upper_map = cluster_std(bt_, wt_, img_f1, tool_, type_)
        ax[ind_[0]].plot(data1, data2, linewidth=0, marker='o', alpha=.5, label=label_)
        ax[ind_[0]].plot(bt_data1, wt_data1, linewidth=0, marker='o', color='purple', alpha=.5, label='Upper cluster')
        #ax[ind_[0]].plot(bt_data2, wt_data2, linewidth=0, marker='o', color='yellow', alpha=.5, label='Lower cluster')
        ax[ind_[0]].plot(bt_data3, wt_data3, linewidth=0, marker='o', color='green', alpha=.5, label='Correlated cluster')

        #ax[ind_[0]].plot(data1, y, color='black', alpha=.5, label='Regression line')
        # ci = 1.96 * np.std(y)/np.mean(y)
        # ax.fill_between(x, (y-ci), (y+ci), color='g', alpha=.9)
        ax[ind_[0]].set_title('')
        ax[ind_[0]].set_xlabel('')
        if ind_[0] == 0: ax[ind_[0]].set_ylabel('Within tool variation (std.)')
        # if ind_[0] == 0: ax[ind_[0]].set_title('Correlation of variances in {}olded maps'.format(type_))
        ax[ind_[0]].set_xlabel('Between tool variation (std.)')
        ax[ind_[0]].set_xlim([-0.15, 5])
        ax[ind_[0]].set_ylim([-0.07, 2.6])
        ax[ind_[0]].legend(facecolor='white', loc='upper right')
        #ax[ind_[0]].set_xscale('log')

        # fraction of clusters
        data2[(data2 == 0) & (data1 == 0)] = np.nan
        all_corr_count = np.count_nonzero(~np.isnan(data2))
        corr_map[np.isnan(bt_) & np.isnan(wt_)] = np.nan
        corr_count = np.count_nonzero(~np.isnan(corr_map))
        upper_map[np.isnan(bt_) & np.isnan(wt_)] = np.nan
        upper_count = np.count_nonzero(~np.isnan(upper_map))
        #print('{}\nall voxels:{}\ncorrelated voxels: {}\nupper voxels: {}\nFraction of green: %{:.1f}\nFraction of purple: %{:.1f}'.format(tool_, all_corr_count, corr_count, upper_count, (corr_count/all_corr_count)*100, (upper_count/all_corr_count)*100))
        print('{}\nFraction of green: %{:.1f}\nFraction of purple: %{:.1f}'.format(tool_, (corr_count/all_corr_count)*100, (upper_count/all_corr_count)*100))
        print('stop')

def plot_corr_variances():
    ### Plot correlation of variances between BT and WT
    for ind1, type_ in enumerate(['unthresh']):
        fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(24, 7))
        #fig.suptitle("Correlation of variances between tool-variability vs. numerical-variability")
        for ind2, pair_ in enumerate(['fsl-spm', 'fsl-afni', 'afni-spm']):
            tool1, tool2 = pair_.split('-')
            fuzzy1 = './figures/map-on-surf/std/fuzzy-{}-{}-std.nii.gz'.format(tool1, type_)
            fuzzy2 = './figures/map-on-surf/std/fuzzy-{}-{}-std.nii.gz'.format(tool2, type_)
            fuzzy12 = './figures/map-on-surf/std/fuzzy-{}-{}-std.nii.gz'.format(pair_, type_)
            tool12 = './figures/map-on-surf/std/{}-{}-std.nii.gz'.format(pair_, type_)

            #plot_pair_corr_variance(tool12, fuzzy1, type_, (ind2, 0), ax, tool_=tool1)
            plot_pair_corr_variance(tool12, fuzzy12, type_, (ind2, 0), ax, tool_=pair_)
            #plot_pair_corr_variance(tool12, fuzzy2, type_, (ind2, 2), ax, tool_=tool2)

            # create new axes on the right and on the top of the current axes
            # divider = make_axes_locatable(ax[ind2])
            # ax_histleft = divider.append_axes("left", 4, pad=0.1, sharex=ax[ind2])
            # ax_histright = divider.append_axes("right", 4, pad=0.1, sharey=ax[ind2])
            # #ax_histleft.xaxis.set_tick_params(labelleft=False)
            # ax_histright.yaxis.set_tick_params(labelleft=False)
            # # X = [i for i in data1 if i != 0.0]
            # # Y = [i for i in data1 if i != 0.0]
            # # ax_histy.hist(Y, bins=300, alpha=.5, orientation='horizontal')
            # plot_pair_corr_variance(tool12, fuzzy1, type_, (ind2, 0), ax_histleft, tool_=tool1, append_='left')
            # plot_pair_corr_variance(tool12, fuzzy2, type_, (ind2, 0), ax_histright, tool_=tool2, append_='right')

        #plt.show()
        plt.savefig('./paper/figures/std-corr-{}-plot.png'.format(type_), bbox_inches='tight')

############# COMPUTES THRESHOLDS

def combine_thresh(tool_results, mca_results):
    ## Thresholded Maps ###
    ## combine activation and deactivation maps ###

    # between tool
    for i in tool_results.keys():
        # Remove NaNs
        n = nib.load(tool_results[i]['exc_set_file'])
        d = n.get_data()
        exc_set_nonan = nib.Nifti1Image(np.nan_to_num(d), n.affine, header=n.header)
        n = nib.load(tool_results[i]['exc_set_file_neg'])
        d = n.get_data()
        exc_set_neg_nonan = nib.Nifti1Image(np.nan_to_num(d), n.affine, header=n.header)
        # Combine activations and deactivations in a single image
        to_display = math_img("img1-img2", img1=exc_set_nonan, img2=exc_set_neg_nonan)
        nib.save(to_display, './figures/map-on-surf/{}.nii.gz'.format(i))

    # between mca samples
    for dic_ in mca_results.keys():
        dic_ = mca_results[dic_]
        tool_ = dic_[1]['exc_set_file'].split('/')[5]
        if tool_ == 'SPM-Octace': tool_ = 'SPM'
        for i in dic_.keys():
            # Remove NaNs
            n = nib.load(dic_[i]['exc_set_file'])
            d = n.get_data()
            exc_set_nonan = nib.Nifti1Image(np.nan_to_num(d), n.affine, header=n.header)
            n = nib.load(dic_[i]['exc_set_file_neg'])
            d = n.get_data()
            exc_set_neg_nonan = nib.Nifti1Image(np.nan_to_num(d), n.affine, header=n.header)
            # Combine activations and deactivations in a single image
            to_display = math_img("img1-img2", img1=exc_set_nonan, img2=exc_set_neg_nonan)
            nib.save(to_display, './figures/map-on-surf/{}_s{}.nii.gz'.format(tool_.lower(), i))


def plot_histo_tstats(tool_results, mca_results):
    fig, ax = plt.subplots(nrows=3,ncols=2,figsize=(16, 16))
    #for ind2_, type_ in enumerate(['stat_file']):
    ind2_ = 0
    type_ = 'stat_file'

    # for pair_ in ['fsl-afni', 'afni-spm', 'fsl-spm']:
    #     t1, t2 = pair_.split('-')
    #     for file_ in ['stat_file']:
    #         mean, diff, md, sd, corr = bland_altman.bland_altman_values(tool_results[t1][file_],
    #                                                                     tool_results[t2][file_], False)
    #         print('{} - {}\nMean of diff: {} \nStd. of diff: {} \nCorr: {}'.format(pair_, file_, md, sd, corr))


    for ind1_, tool_ in enumerate(['fsl', 'spm', 'afni']):
        f_ = tool_results[tool_][type_]
        f_img = nib.load(f_)
        f_data = np.reshape(f_img.get_fdata(), -1)
        X = [i for i in f_data if i != 0.0]
        ax[ind1_, ind2_].hist(X, bins=300, alpha=.5, range=(-5,6), label=tool_.upper(), color='b')

        pos_thresh = get_min(tool_results[tool_]['exc_set_file'])
        neg_thresh = -(get_min(tool_results[tool_]['exc_set_file_neg']))
        ax[ind1_, ind2_].axvline(pos_thresh, label=f'T=\u00B1{pos_thresh:.3f}')
        ax[ind1_, ind2_].axvline(neg_thresh)

        ax[ind1_, ind2_].set_ylim([0,3200])
        #ax[ind1_, ind2_].set_yscale('symlog')
        ax[ind1_, ind2_].set_xlim([-6,6])
        ax[ind1_, ind2_].set_title('')
        ax[ind1_, ind2_].set_xlabel('')
        ax[ind1_, ind2_].set_ylabel('# of voxels')
        #if ind2_ == 0: ax[ind1_, ind2_].set_title('Histogram of t stats in {}'.format(tool_))
        if ind1_ == 2: ax[ind1_, ind2_].set_xlabel('T-statistics')
        ax[ind1_, ind2_].legend(facecolor='white')
        ax[ind1_, ind2_].yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True))
        ax[ind1_, ind2_].ticklabel_format(axis="y", style="sci", scilimits=(0,0))

    ind2_ = 1
    for ind1_, tool_ in enumerate(['fsl', 'spm', 'afni']):
        for i in range(1,4):
            f_ = mca_results[tool_][i][type_]
            f_img = nib.load(f_)
            f_data = np.reshape(f_img.get_fdata(), -1)
            X = [i for i in f_data if i != 0.0]
            ax[ind1_, ind2_].hist(X, bins=300, alpha=.5, range=(-5,6), label='{}{}'.format(tool_.upper(), i))

            pos_thresh = get_min(mca_results[tool_][i]['exc_set_file'])
            neg_thresh = -(get_min(mca_results[tool_][i]['exc_set_file_neg']))
            ax[ind1_, ind2_].axvline(pos_thresh, label=f'T=\u00B1{pos_thresh:.3f}')
            ax[ind1_, ind2_].axvline(neg_thresh)

            ax[ind1_, ind2_].set_ylim([0,3200])
            #ax[ind1_, ind2_].set_yscale('symlog')
            ax[ind1_, ind2_].set_xlim([-6,6])
            ax[ind1_, ind2_].set_title('')
            ax[ind1_, ind2_].set_xlabel('')
            ax[ind1_, ind2_].set_ylabel('# of voxels')
            #if ind2_ == 0: ax[ind1_, ind2_].set_title('Histogram of t stats in {}'.format(tool_))
            if ind1_ == 2: ax[ind1_, ind2_].set_xlabel('T-statistics')
            ax[ind1_, ind2_].legend(facecolor='white')
            ax[ind1_, ind2_].yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True))
            ax[ind1_, ind2_].ticklabel_format(axis="y", style="sci", scilimits=(0,0))

    #plt.show()
    plt.savefig('./paper/figures/tstats-hist-plot.png', bbox_inches='tight')


def get_min(file_):
    f_img = nib.load(file_)
    f_data = f_img.get_fdata()
    f_data = np.reshape(np.nan_to_num(f_data), -1)
    X = [i for i in f_data if i != 0.0]
    return(min(X))

############# COMPUTES dICES


def dump_variable(var_, name_):
    with open('./data/{}.pkl'.format(str(name_)),'wb') as f:
        pkl.dump(var_, f)


def load_variable(name_):
    with open('./data/{}.pkl'.format(name_),'rb') as fr:
        return pkl.load(fr)


def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]


def compute_dice(data1_file, data2_file):

    # Load nifti images
    data1_img = nib.load(data1_file)
    data2_img = nib.load(data2_file)
    # Load data from images
    data2 = data2_img.get_data()
    data1 = data1_img.get_data()
    # Get asbolute values (positive and negative blobs are of interest)
    data2 = np.absolute(data2)
    data1 = np.absolute(data1)

    # Resample data2 on data1 using nearest neighbours
    data2_resl_img = resample_from_to(data2_img, data1_img, order=0)
    data2_res = data2_resl_img.get_data()
    data2_res = np.absolute(data2_res)

    # Masking (compute Dice using intersection of both masks)
    # bg_ = np.logical_or(np.isnan(data1), np.isnan(data2_res))

    # Binarizing data
    data1 = np.nan_to_num(data1)
    data1[data1>0] = 1
    data2_res = np.nan_to_num(data2_res)
    data2_res[data2_res>0] = 1

    num_activated_1 = np.sum(data1 > 0)
    num_activated_res_2 = np.sum(data2_res>0)

    # Masking intersections
    # data1[bg_] = 0
    # data2_res[bg_] = 0

    # Vectorize
    data1 = np.reshape(data1, -1)
    data2_res = np.reshape(data2_res, -1)

    # similarity = 1.0 - dissimilarity
    dice_score = 1-scipy.spatial.distance.dice(data1>0, data2_res>0)

    return (dice_score, num_activated_1, num_activated_res_2)


def keep_roi(stat_img, reg_, image_parc, filename=None):
    parc_img = nib.load(image_parc)
    data_img = nib.load(stat_img)
    # Resample parcellation on data1 using nearest neighbours
    parc_img_res = resample_from_to(parc_img, data_img, order=0)
    parc_data_res = parc_img_res.get_fdata(dtype=np.float32)
    colls = np.where(parc_data_res != reg_)
    parc_data_res[colls] = np.nan
    # data_img_nan = nib.Nifti1Image(parc_data_res, img_.affine, header=img_.header)
    # nib.save(data_img_nan, 'parce_region{}.nii.gz'.format(reg_))

    data_orig = data_img.get_data()
    # If there are NaNs in data_file remove them (to mask using parcelled data only)
    data_orig = np.nan_to_num(data_orig)
    # Replace background by NaNs
    data_nan = data_orig.astype(float)
    data_nan[np.isnan(parc_data_res)] = np.nan

    # Save as image
    data_img_nan = nib.Nifti1Image(data_nan, data_img.get_affine())
    if filename is None:
        filename = stat_img.replace('.nii', '_nan.nii')
    nib.save(data_img_nan, filename)

    return(filename)


def get_dice_values(regions_txt, image_parc, tool_results, mca_results):
    # read parcellation data
    parc_img = nib.load(image_parc)
    parc_data = parc_img.get_fdata(dtype=np.float32)

    regions = {}
    with open(regions_txt) as f:
        for line in f:
            (key, val) = line.split()
            regions[int(key)+180] = val
            regions[int(key)] = 'R_' +  '_'.join(val.split('_')[1:])

    dices_ = {}
    #for act_ in ['exc_set_file', 'exc_set_file_neg', 'act_deact', 'stat_file']:
    for act_ in ['act_deact']:
        masked_regions = {}
        for r in regions.keys():
            colls = np.where(parc_data == r)
            if regions[r] not in dices_.keys():
                dices_[regions[r]] = {}
                dices_[regions[r]]['size'] = len(colls[0])
                dices_[regions[r]][act_] = {}
                dices_[regions[r]][act_]['tool'] = {}
                dices_[regions[r]][act_]['mca'] = {}
            else:
                dices_[regions[r]][act_] = {}
                dices_[regions[r]][act_]['tool'] = {}
                dices_[regions[r]][act_]['mca'] = {}

            masked_regions['fsl'] = keep_roi(tool_results['fsl'][act_], r, image_parc)
            masked_regions['afni'] = keep_roi(tool_results['afni'][act_], r, image_parc)
            masked_regions['spm'] = keep_roi(tool_results['spm'][act_], r, image_parc)

            # dices_[regions[r]][act_]['tool']['fsl-afni'] = dice.sorrenson_dice(masked_regions['fsl'], masked_regions['afni'])[0]
            dices_[regions[r]][act_]['tool']['fsl-afni'] = compute_dice(masked_regions['fsl'], masked_regions['afni'])[0]
            # dices_[regions[r]][act_]['tool']['fsl-spm'] = dice.sorrenson_dice(masked_regions['fsl'], masked_regions['spm'])[0]
            dices_[regions[r]][act_]['tool']['fsl-spm'] = compute_dice(masked_regions['fsl'], masked_regions['spm'])[0]
            # dices_[regions[r]][act_]['tool']['afni-spm'] = dice.sorrenson_dice(masked_regions['afni'], masked_regions['spm'])[0]
            dices_[regions[r]][act_]['tool']['afni-spm'] = compute_dice(masked_regions['afni'], masked_regions['spm'])[0]
            # dices = (dice_res_1, dark_dice_1[1], dark_dice_2[1], num_activated_1, num_activated_2)

            for tool_ in mca_results.keys():
                masked_regions['{}1'.format(tool_)] = keep_roi(mca_results[tool_][1][act_], r, image_parc)#, '{}1'.format(tool_))
                masked_regions['{}2'.format(tool_)] = keep_roi(mca_results[tool_][2][act_], r, image_parc)#, '{}2'.format(tool_))
                masked_regions['{}3'.format(tool_)] = keep_roi(mca_results[tool_][3][act_], r, image_parc)#, '{}3'.format(tool_))

                # dices_[regions[r]][act_]['mca']['{}1'.format(tool_)] = dice.sorrenson_dice(masked_regions['{}1'.format(tool_)], masked_regions['{}2'.format(tool_)])[0]
                dices_[regions[r]][act_]['mca']['{}1'.format(tool_)] = compute_dice(masked_regions['{}1'.format(tool_)], masked_regions['{}2'.format(tool_)])[0]
                # dices_[regions[r]][act_]['mca']['{}2'.format(tool_)] = dice.sorrenson_dice(masked_regions['{}1'.format(tool_)], masked_regions['{}3'.format(tool_)])[0]
                dices_[regions[r]][act_]['mca']['{}2'.format(tool_)] = compute_dice(masked_regions['{}1'.format(tool_)], masked_regions['{}3'.format(tool_)])[0]
                # dices_[regions[r]][act_]['mca']['{}3'.format(tool_)] = dice.sorrenson_dice(masked_regions['{}2'.format(tool_)], masked_regions['{}3'.format(tool_)])[0]
                dices_[regions[r]][act_]['mca']['{}3'.format(tool_)] = compute_dice(masked_regions['{}2'.format(tool_)], masked_regions['{}3'.format(tool_)])[0]

    dump_variable(dices_, retrieve_name(dices_)[0])
    return dices_


def plot_dices(dices_):

    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(16, 8))
    marker_ = ['o', '*', 'v', '>', '1', '2', '3', '4', 'P']
    colors = ['red', 'blue', 'green']
    for ind1_, act_ in enumerate(['act_deact']):
        for ind_, tool_ in enumerate(['fsl-afni', 'fsl-spm', 'afni-spm']):
            tool_list = []
            mca_list = []
            for reg_ in dices_.keys():
                tool_list.append(dices_[reg_][act_]['tool'][tool_])
                f1, f2 = tool_.split('-')
                mca_mean = []
                for i in [1, 2, 3]:
                    mca_mean.append(dices_[reg_][act_]['mca']['{}{}'.format(f1, i)])
                    mca_mean.append(dices_[reg_][act_]['mca']['{}{}'.format(f2, i)])
                mca_list.append(mean(np.nan_to_num(mca_mean)))

            # Get region sizes
            tool_list = np.nan_to_num(tool_list)
            r_sizes = []
            r_name = []
            for i in range(len(tool_list)):
                l_ = list(dices_.keys())
                s = dices_[l_[i]]['size']
                r_sizes.append(s)
                r_name.append(l_[i])
    
            # Plot Normalize dice values by region size
            tool_norm = np.array(tool_list)*np.array(r_sizes)
            mca_norm = np.array(mca_list)*np.array(r_sizes)
            # Normalization between [0-1] = x -xmin/ xmax – xmin
            tool_norm = (tool_norm - np.min(tool_norm)) / (np.max(tool_norm) - np.min(tool_norm))
            mca_norm = (mca_norm - np.min(mca_norm)) / (np.max(mca_norm) - np.min(mca_norm))

            # regions that BT=0 and WT>0.93
            x = np.where((tool_norm == 0) & (mca_norm > 0.93) , r_name, None)
            res = [i for i in x if i]
            print("BT only var regions in {}:\n{}".format(tool_, res))
                
            # Compute regression line without zero values
            nonz_tool_norm = []
            nonz_mca_norm = []
            for i in range(len(tool_norm)):
                if tool_norm[i] != 0  and mca_norm[i] != 0:
                    nonz_tool_norm.append(tool_norm[i])
                    nonz_mca_norm.append(mca_norm[i])

            slope_norm, intercept_norm, r_norm, p_norm, stderr = scipy.stats.linregress(nonz_tool_norm, nonz_mca_norm)
            line_norm = f'Regression line: y={intercept_norm:.2f}+{slope_norm:.2f}x, r={r_norm:.2f}, p={p_norm:.10f}'
            y_norm = intercept_norm + slope_norm * np.array(tool_list)

            ax.plot(tool_norm, mca_norm, linewidth=0, alpha=.5, color=colors[ind_], marker='o', label='{}'.format(tool_.upper()))
            ax.plot(tool_list, y_norm, color=colors[ind_], alpha=.7, label=line_norm)
            ax.set_xlabel('Normalized Dice scores in BT')
            ax.set_ylabel('Normalized Dice scores in WT')
            # ax.set_title('Normalized Dice scores from thresholded maps')
            ax.legend()


    #plt.show()
    plt.savefig('./paper/figures/dices_corr.png', bbox_inches='tight')

############# COMPUTES STATS

def compute_stats(data1_file, data2_file):
    # Load nifti images
    data1_img = nib.load(data1_file)
    data2_img = nib.load(data2_file)
    # Resample data2 on data1 using nearest neighbours
    data2_img_res = resample_from_to(data2_img, data1_img, order=0)
    
    # Load data from images
    data1 = data1_img.get_data()
    data2 = data2_img_res.get_data()
    # Vectorise input data
    data1 = np.reshape(data1, -1)
    data2 = np.reshape(data2, -1)

    diff_ = np.abs(data1 - data2)  # Absolute difference between data1 and data2
    mean_ = np.mean(diff_)                   # Mean of the difference
    # std_ = np.std(diff_, axis=0)            # Standard deviation of the difference
    # corr_ = np.corrcoef(data1, data2)[0,1] #  The correlation coefficient 
    return mean_


def print_stats(tool_results, mca_results):
    # Compute Mean of diff over whole image (no mask for bg)
    for type_ in ['stat_file', 'act_deact']: # 'stat_file': 'unthresh' and 'act_deact': 'thresh'
        for pair_ in ['fsl-spm', 'fsl-afni', 'afni-spm']:
            t1, t2 = pair_.split('-')
            f1 = tool_results[t1][type_]
            f2 = tool_results[t2][type_]
            bt_mean_diff = compute_stats(f1, f2)

            # Fuzzy samples
            means_ = []
            means_.append(compute_stats(mca_results[t1][1][type_], mca_results[t1][2][type_]))
            means_.append(compute_stats(mca_results[t1][1][type_], mca_results[t1][3][type_]))
            means_.append(compute_stats(mca_results[t1][2][type_], mca_results[t1][3][type_]))
            means_.append(compute_stats(mca_results[t2][1][type_], mca_results[t2][2][type_]))
            means_.append(compute_stats(mca_results[t2][1][type_], mca_results[t2][3][type_]))
            means_.append(compute_stats(mca_results[t2][2][type_], mca_results[t2][3][type_]))
            wt_mean_diff = mean(means_)
            print('Mean of diff of Tstats in {} {}:\nBT {}\nWT {}'.format(pair_, type_, bt_mean_diff, wt_mean_diff))

    # Compute Std of differences (ignore NaNs/zeros as bg)    
    path_ = './figures/map-on-surf/std/'
    for type_ in ['thresh', 'unthresh']:
        for pair_ in ['fsl-spm', 'fsl-afni', 'afni-spm']:
            img1_ = nib.load('{}{}-{}-std.nii.gz'.format(path_, pair_, type_))
            img2_ = nib.load('{}fuzzy-{}-{}-std.nii.gz'.format(path_, pair_, type_))
            # img1_data = np.nan_to_num(img1_.get_fdata())
            # img2_data = np.nan_to_num(img2_.get_fdata())
            bt_std_data = img1_.get_fdata()
            wt_std_data = img2_.get_fdata()
            bt_std_mean = np.nanmean(bt_std_data)
            wt_std_mean = np.nanmean(wt_std_data)
            print('Std of diff of Tstats in {} {}:\nBT {}\nWT {}'.format(pair_, type_, bt_std_mean, wt_std_mean))
            
            # Compute SE: n=2 for bt and n=6 for wt
            bt_se_data = bt_std_data/np.sqrt(2)
            wt_se_data = wt_std_data/np.sqrt(6)
            bt_se_mean = np.nanmean(bt_se_data)
            wt_se_mean = np.nanmean(wt_se_data)
            print('SE of diff of Tstats in {} {}:\nBT {}\nWT {}'.format(pair_, type_, bt_se_mean, wt_se_mean))

            print("stop")


def main(args=None):

    tool_results = {}
    tool_results['fsl'] = {}
    tool_results['fsl']['exc_set_file'] = './results/ds000001/tools/FSL/thresh_zstat1.nii.gz'
    tool_results['fsl']['exc_set_file_neg'] = './results/ds000001/tools/FSL/thresh_zstat2.nii.gz'
    tool_results['fsl']['stat_file'] = './results/ds000001/tools/FSL/tstat1.nii.gz'
    tool_results['fsl']['act_deact'] = './figures/map-on-surf/fsl.nii.gz'
    tool_results['spm'] = {}
    tool_results['spm']['exc_set_file'] = './results/ds000001/tools/SPM/Octave/spm_exc_set.nii.gz'
    tool_results['spm']['exc_set_file_neg'] = './results/ds000001/tools/SPM/Octave/spm_exc_set_neg.nii.gz'
    tool_results['spm']['stat_file'] = './results/ds000001/tools/SPM/Octave/spm_stat.nii.gz'
    # tool_results['spm']['exc_set_file'] = './results/ds000001/tools/SPM/Matlab2016b/spm_exc_set.nii.gz'
    # tool_results['spm']['exc_set_file_neg'] = './results/ds000001/tools/SPM/Matlab2016b/spm_exc_set_neg.nii.gz'
    # tool_results['spm']['stat_file'] = './results/ds000001/tools/SPM/Matlab2016b/spm_stat.nii.gz'
    tool_results['spm']['act_deact'] = './figures/map-on-surf/spm.nii.gz'
    tool_results['afni'] = {}
    tool_results['afni']['exc_set_file'] = './results/ds000001/tools/AFNI/Positive_clustered_t_stat.nii.gz'
    tool_results['afni']['exc_set_file_neg'] = './results/ds000001/tools/AFNI/Negative_clustered_t_stat.nii.gz'
    tool_results['afni']['stat_file'] = './results/ds000001/tools/AFNI/3dMEMA_result_t_stat_masked.nii.gz'
    tool_results['afni']['act_deact'] = './figures/map-on-surf/afni.nii.gz'

    fsl_mca = {}
    fsl_mca[1] = {}
    fsl_mca[1]['exc_set_file'] = "./results/ds000001/fuzzy/p53/FSL/run1/thresh_zstat1_53_run1.nii.gz"
    fsl_mca[1]['exc_set_file_neg'] = "./results/ds000001/fuzzy/p53/FSL/run1/thresh_zstat2_53_run1.nii.gz"
    fsl_mca[1]['stat_file'] = "./results/ds000001/fuzzy/p53/FSL/run1/tstat1_53_run1.nii.gz"
    fsl_mca[1]['act_deact'] = "./figures/map-on-surf/fsl_s1.nii.gz"
    fsl_mca[2] = {}
    fsl_mca[2]['exc_set_file'] = "./results/ds000001/fuzzy/p53/FSL/run2/thresh_zstat1_53_run2.nii.gz"
    fsl_mca[2]['exc_set_file_neg'] = "./results/ds000001/fuzzy/p53/FSL/run2/thresh_zstat2_53_run2.nii.gz"
    fsl_mca[2]['stat_file'] = "./results/ds000001/fuzzy/p53/FSL/run2/tstat1_53_run2.nii.gz"
    fsl_mca[2]['act_deact'] = "./figures/map-on-surf/fsl_s2.nii.gz"
    fsl_mca[3] = {}
    fsl_mca[3]['exc_set_file'] = "./results/ds000001/fuzzy/p53/FSL/run3/thresh_zstat1_53_run3.nii.gz"
    fsl_mca[3]['exc_set_file_neg'] = "./results/ds000001/fuzzy/p53/FSL/run3/thresh_zstat2_53_run3.nii.gz"
    fsl_mca[3]['stat_file'] = "./results/ds000001/fuzzy/p53/FSL/run3/tstat1_53_run3.nii.gz"
    fsl_mca[3]['act_deact'] = "./figures/map-on-surf/fsl_s3.nii.gz"

    spm_mca = {}
    spm_mca[1] = {}
    spm_mca[1]['exc_set_file'] = "./results/ds000001/fuzzy/p53/SPM-Octace/run1/spm_exc_set.nii.gz"
    spm_mca[1]['exc_set_file_neg'] = "./results/ds000001/fuzzy/p53/SPM-Octace/run1/spm_exc_set_neg.nii.gz"
    spm_mca[1]['stat_file'] = "./results/ds000001/fuzzy/p53/SPM-Octace/run1/spm_stat.nii.gz"
    spm_mca[1]['act_deact'] = "./figures/map-on-surf/spm_s1.nii.gz"
    spm_mca[2] = {}
    spm_mca[2]['exc_set_file'] = "./results/ds000001/fuzzy/p53/SPM-Octace/run2/spm_exc_set.nii.gz"
    spm_mca[2]['exc_set_file_neg'] = "./results/ds000001/fuzzy/p53/SPM-Octace/run2/spm_exc_set_neg.nii.gz"
    spm_mca[2]['stat_file'] = "./results/ds000001/fuzzy/p53/SPM-Octace/run2/spm_stat.nii.gz"
    spm_mca[2]['act_deact'] = "./figures/map-on-surf/spm_s2.nii.gz"
    spm_mca[3] = {}
    spm_mca[3]['exc_set_file'] = "./results/ds000001/fuzzy/p53/SPM-Octace/run3/spm_exc_set.nii.gz"
    spm_mca[3]['exc_set_file_neg'] = "./results/ds000001/fuzzy/p53/SPM-Octace/run3/spm_exc_set_neg.nii.gz"
    spm_mca[3]['stat_file'] = "./results/ds000001/fuzzy/p53/SPM-Octace/run3/spm_stat.nii.gz"
    spm_mca[3]['act_deact'] = "./figures/map-on-surf/spm_s3.nii.gz"

    afni_mca = {}
    afni_mca[1] = {}
    afni_mca[1]['exc_set_file'] = "./results/ds000001/fuzzy/p53/AFNI/run1/Positive_clustered_t_stat.nii.gz"
    afni_mca[1]['exc_set_file_neg'] = "./results/ds000001/fuzzy/p53/AFNI/run1/Negative_clustered_t_stat.nii.gz"
    afni_mca[1]['stat_file'] = "./results/ds000001/fuzzy/p53/AFNI/run1/3dMEMA_result_t_stat_masked.nii.gz"
    afni_mca[1]['act_deact'] = "./figures/map-on-surf/afni_s1.nii.gz"
    afni_mca[2] = {}
    afni_mca[2]['exc_set_file'] = "./results/ds000001/fuzzy/p53/AFNI/run2/Positive_clustered_t_stat.nii.gz"
    afni_mca[2]['exc_set_file_neg'] = "./results/ds000001/fuzzy/p53/AFNI/run2/Negative_clustered_t_stat.nii.gz"
    afni_mca[2]['stat_file'] = "./results/ds000001/fuzzy/p53/AFNI/run2/3dMEMA_result_t_stat_masked.nii.gz"
    afni_mca[2]['act_deact'] = "./figures/map-on-surf/afni_s2.nii.gz"
    afni_mca[3] = {}
    afni_mca[3]['exc_set_file'] = "./results/ds000001/fuzzy/p53/AFNI/run3/Positive_clustered_t_stat.nii.gz"
    afni_mca[3]['exc_set_file_neg'] = "./results/ds000001/fuzzy/p53/AFNI/run3/Negative_clustered_t_stat.nii.gz"
    afni_mca[3]['stat_file'] = "./results/ds000001/fuzzy/p53/AFNI/run3/3dMEMA_result_t_stat_masked.nii.gz"
    afni_mca[3]['act_deact'] = "./figures/map-on-surf/afni_s3.nii.gz"

    mca_results = {}
    mca_results['fsl'] = {}
    mca_results['fsl'] = fsl_mca
    mca_results['spm'] = {}
    mca_results['spm'] = spm_mca
    mca_results['afni'] = {}
    mca_results['afni'] = afni_mca
    # spm_parcel = './HCP-MMP1_on_MNI152_spm.nii.gz'
    # fsl_parcel = './HCP-MMP1_on_MNI152_fsl.nii.gz'
    # afni_parcel = './HCP-MMP1_on_MNI152_afni.nii.gz'

    ### combine activation and deactivation maps ###
    #combine_thresh(tool_results, mca_results)

    ### Compute stats
    #print_stats(tool_results, mca_results)

    ### Variances between tools
    #var_between_tool(tool_results)
    ### Variances between fuzzy samples
    #var_between_fuzzy(mca_results)
    ### get the ratio between BT and WT std images
    #get_ratio()

    ### Plot correlation of variances between BT and FL
    #plot_corr_variances()

    ### Plot histogram of t-stats (thresholds)
    #plot_histo_tstats(tool_results, mca_results)

    ### Compute Dice scores and then plot
    # image_parc = './MNI-parcellation/HCP-MMP1_on_MNI152_ICBM2009a_nlin.nii.gz'
    # image_parc = './MNI-parcellation/HCPMMP1_on_MNI152_ICBM2009a_nlin_resampled.splitLR.nii.gz'
    # regions_txt = './MNI-parcellation/HCP-MMP1_on_MNI152_ICBM2009a_nlin.txt'
    # if os.path.exists('./data/dices_.pkl'):
    #     dices_ = load_variable('dices_')
    #     plot_dices(dices_)
    # else:
    #     dices_ = get_dice_values(regions_txt, image_parc, tool_results, mca_results)
    #     plot_dices(dices_)


if __name__ == '__main__':
    main()