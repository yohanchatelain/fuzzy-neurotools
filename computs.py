#!/usr/bin/env python3

from nibabel.processing import resample_from_to
import scipy
import scipy.spatial
import nibabel as nib
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle as pkl
import inspect
from nilearn.image import math_img
from lib import dice
from statistics import mean
import scipy.stats as stat
from scipy.signal import resample
import os


############# Global nearest precision

def calculate_rmse(im1, im2):
    """Computing the RMSE between two images."""
    img1 = np.nan_to_num(im1.get_fdata())
    img2 = np.nan_to_num(im2.get_fdata())

    if img1.shape != img2.shape:
        im1 = resample_from_to(im1, im2, order=0)
        img1 = np.nan_to_num(im1.get_fdata())

    bg_ = np.where((np.isnan(im1.get_fdata())) & (np.isnan(im2.get_fdata())), False, True)
    img1 = img1[bg_]
    img2 = img2[bg_]

    return np.sqrt(np.mean((img1-img2)**2, dtype=np.float64), dtype=np.float64)


def global_nearest_precision():
    rmse_ = {}
    all_rmse = {}
    global_average_rmse = []

    for p in range(11, 54, 2):
        rmse_[p] = 0
        all_rmse[str(p)] = []
        for s in ['fsl-afni', 'fsl-spm', 'afni-spm']:
            bt_ = nib.load('./data/std/{}-unthresh-std.nii.gz'.format(s))
            # bt_ = bt_.get_fdata()
            wt_fsl = nib.load('./data/std/FL-FSL/p{}_fsl_unthresh_std.nii.gz'.format(p))
            # wt_fsl = wt_fsl.get_fdata()
            rmse_value = calculate_rmse(bt_, wt_fsl)
            all_rmse[str(p)].append(rmse_value)
            rmse_[p] += rmse_value

    all_rmse = np.transpose(np.array(list(all_rmse.values())))

    return min(rmse_, key=rmse_.get), all_rmse


def plot_rmse_nearest(all_rmse):
    import matplotlib.cm as cm

    fig = plt.figure(figsize=(15,10))
    colors = ['red', 'green', 'blue']
    tool_pair = ['fsl-afni', 'fsl-spm', 'afni-spm']
    for s, rmse_ in enumerate(all_rmse):
        #type_ = ""
        #if s+1 in i2T1w: type_ += "*"
        #if s+1 in i2T2w: type_ += "$\dag$"
        plt.plot(range(11,54, 2), rmse_, marker='x', color=colors[s], alpha=0.8, label=tool_pair[s].upper())

    average = np.mean(all_rmse, axis=0, dtype=np.float64)
    p1 = plt.plot(range(11,54, 2), average, marker='o', linewidth=2, color='black', label='Average') #, color='grey', alpha=0.5)
    plt.xticks(range(11,54, 2), fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Virtual precision (bits)', fontsize=22)
    plt.ylabel('RMSE (std)', fontsize=22)

    plt.legend(fontsize=16)
    # handles, labels = plt.gca().get_legend_handles_labels()
    # order = color_sbj_order
    # legend2 = plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=12, bbox_to_anchor=(1, .96), title="Subjects")

    # legend1 = plt.legend(p1, ["Average"], bbox_to_anchor=(1.105,1))# bbox_to_anchor=(1.105,0.25) #loc=1
    # plt.gca().add_artist(legend1)
    # plt.gca().add_artist(legend2)

    plt.savefig('./paper/figures/rmse-precisions.png', bbox_inches='tight', facecolor='w', transparent=False)


############# COMPUTE and PLOT STD. DEV.

def compute_std_WT(path_):
    for p in range (11, 54, 2):
        s1 = os.path.join(path_, 'p{}/FSL/run1/tstat1.nii.gz'.format(p))
        s2 = os.path.join(path_, 'p{}/FSL/run2/tstat1.nii.gz'.format(p))
        s3 = os.path.join(path_, 'p{}/FSL/run3/tstat1.nii.gz'.format(p))
        f1 = nib.load(s1)
        f1_data = f1.get_fdata()
        f2 = nib.load(s2)
        f2_data = f2.get_fdata()
        f3 = nib.load(s3)
        f3_data = f3.get_fdata()
        img_concat = nib.funcs.concat_images([f1, f2, f3], check_affines=True, axis=None)
        img_std = np.std(img_concat.get_fdata(), axis=3)
        # nan bg images
        img_std = np.where((f1_data == 0) & (f2_data == 0) & (f3_data == 0), np.nan, img_std)
        nft_img_std = nib.Nifti1Image(img_std, f1.affine, header=f1.header)
        nib.save(nft_img_std, os.path.join('data/std/FL-FSL/', 'p{}_fsl_unthresh_std.nii.gz'.format(p)))


def compute_var(f1, f2, file_name=None, f3=None):
    # Compute std. in BT
    f1 = nib.load(f1)
    img_concat = nib.funcs.concat_images([f1, f2], check_affines=True, axis=None)
    img_std = np.std(img_concat.get_fdata(), axis=3)
    f1_data = f1.get_fdata()
    if type(f2) == str : f2 = nib.load(f2)
    f2_data = f2.get_fdata()
    # activated regions have nonzero values
    img_std = np.where((f1_data == 0) & (f2_data == 0), np.nan, img_std)
    nft_img = nib.Nifti1Image(img_std, f1.affine, header=f1.header)
    nib.save(nft_img, os.path.join('data/std/', '{}-std.nii.gz'.format(file_name)))

    # Compute variance and std. in WT
    if f3 is not None:
        img_concat = nib.funcs.concat_images([f1, f2, f3], check_affines=True, axis=None)       
        img_std = np.std(img_concat.get_fdata(), axis=3)

        if type(f3) == str : f3 = nib.load(f3)
        f3_data = f3.get_fdata()
        img_std = np.where((f1_data == 0) & (f2_data == 0) & (f3_data == 0), np.nan, img_std)

        nft_img_std = nib.Nifti1Image(img_std, f1.affine, header=f1.header)
        nib.save(nft_img_std, os.path.join('data/std/', '{}-std.nii.gz'.format(file_name)))

        img_var = np.var(img_concat.get_fdata(), axis=3)
        img_var = np.where((f1_data == 0) & (f2_data == 0) & (f3_data == 0), np.nan, img_var)
        nft_img = nib.Nifti1Image(img_var, f1.affine, header=f1.header)
        return nft_img


def combine_var(f1, f2, meta_, file_name):
    var_f2_res = resample_from_to(f2, f1, order=0)
    # to combine two image variances, we use: var(x+y) = var(x) + var(y) + 2*cov(x,y)
    # and since the correlation between two arrays are so weak, we droped `2*cov(x,y)` from the formula
    f1t = f1.get_fdata()
    f2t = var_f2_res.get_fdata()
    combine_var = np.nan_to_num(f1.get_fdata()) + np.nan_to_num(var_f2_res.get_fdata())
    std_ = np.sqrt(combine_var)
    # nan bg images
    std_ = np.where((np.isnan(f1t)) & (np.isnan(f2t)), np.nan, std_)
    nft_img = nib.Nifti1Image(std_, meta_.affine, header=meta_.header)
    nib.save(nft_img, os.path.join('data/std/', '{}-std.nii.gz'.format(file_name)))


def var_between_tool(tool_results):
    ## thresholded group-level
    fsl_ = tool_results['fsl']['act_deact']
    afni_ = tool_results['afni']['act_deact']
    spm_ = tool_results['spm']['act_deact']
    # resampling first image on second image
    spm_res = resample_from_to(nib.load(spm_), nib.load(fsl_), order=0)
    afni_res = resample_from_to(nib.load(afni_), nib.load(fsl_), order=0)
    # compute variances
    compute_var(fsl_, afni_res, 'fsl-afni-thresh', f3=None)
    compute_var(fsl_, spm_res, 'fsl-spm-thresh', f3=None)
    spm_res = resample_from_to(nib.load(spm_), nib.load(afni_), order=0)
    compute_var(afni_, spm_res, 'afni-spm-thresh', f3=None)

    # unthresholded group-level
    fsl_ = tool_results['fsl']['stat_file']
    afni_ = tool_results['afni']['stat_file']
    spm_ = tool_results['spm']['stat_file']
    # resampling first image on second image
    spm_res = resample_from_to(nib.load(spm_), nib.load(fsl_), order=0)
    afni_res = resample_from_to(nib.load(afni_), nib.load(fsl_), order=0)
    # compute variances
    compute_var(fsl_, afni_res, 'fsl-afni-unthresh', f3=None)
    compute_var(fsl_, spm_res, 'fsl-spm-unthresh', f3=None)
    spm_res = resample_from_to(nib.load(spm_), nib.load(afni_), order=0)
    compute_var(afni_, spm_res, 'afni-spm-unthresh', f3=None)

    ## unthresholded subject-level
    for i in range(1, 17):
        fsl_ = tool_results['fsl']['SBJ'].replace('NUM', '%.2d' % i )
        afni_ = tool_results['afni']['SBJ'].replace('NUM', '%.2d' % i )
        spm_ = tool_results['spm']['SBJ'].replace('NUM', '%.2d' % i )
        # resampling first image on second image
        spm_res = resample_from_to(nib.load(spm_), nib.load(fsl_), order=0)
        afni_res = resample_from_to(nib.load(afni_), nib.load(fsl_), order=0)
        # compute variances
        compute_var(fsl_, afni_res, 'sbj{}-fsl-afni-unthresh'.format('%.2d' % i), f3=None)
        compute_var(fsl_, spm_res, 'sbj{}-fsl-spm-unthresh'.format('%.2d' % i), f3=None)
        spm_res = resample_from_to(nib.load(spm_), nib.load(afni_), order=0)
        compute_var(afni_, spm_res, 'sbj{}-afni-spm-unthresh'.format('%.2d' % i), f3=None)


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
    fsl_var = compute_var(fsl_1, fsl_2, 'fsl-thresh', f3=fsl_3)
    afni_var = compute_var(afni_1, afni_2, 'afni-thresh', f3=afni_3)
    spm_var = compute_var(spm_1, spm_2, 'spm-thresh', f3=spm_3)
    # combine variances
    combine_var(fsl_var, afni_var, nib.load(fsl_1), 'fuzzy-fsl-afni-thresh')
    combine_var(fsl_var, spm_var, nib.load(fsl_1), 'fuzzy-fsl-spm-thresh')
    combine_var(afni_var, spm_var, nib.load(afni_1), 'fuzzy-afni-spm-thresh')

    # unthresholded group-level
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
    fsl_var = compute_var(fsl_1, fsl_2, 'fsl-unthresh', f3=fsl_3)
    afni_var = compute_var(afni_1, afni_2, 'afni-unthresh', f3=afni_3)
    spm_var = compute_var(spm_1, spm_2, 'spm-unthresh', f3=spm_3)
    # combine variances
    combine_var(fsl_var, afni_var, nib.load(fsl_1), 'fuzzy-fsl-afni-unthresh')
    combine_var(fsl_var, spm_var, nib.load(fsl_1), 'fuzzy-fsl-spm-unthresh')
    combine_var(afni_var, spm_var, nib.load(afni_1), 'fuzzy-afni-spm-unthresh')

    # unthresholded subject-level
    for i in range(1, 17):
        fsl_1 = mca_results['fsl'][1]['SBJ'].replace('NUM', '%.2d' % i )
        fsl_2 = mca_results['fsl'][2]['SBJ'].replace('NUM', '%.2d' % i )
        fsl_3 = mca_results['fsl'][3]['SBJ'].replace('NUM', '%.2d' % i )
        afni_1 = mca_results['afni'][1]['SBJ'].replace('NUM', '%.2d' % i )
        afni_2 = mca_results['afni'][2]['SBJ'].replace('NUM', '%.2d' % i )
        afni_3 = mca_results['afni'][3]['SBJ'].replace('NUM', '%.2d' % i )
        spm_1 = mca_results['spm'][1]['SBJ'].replace('NUM', '%.2d' % i )
        spm_2 = mca_results['spm'][2]['SBJ'].replace('NUM', '%.2d' % i )
        spm_3 = mca_results['spm'][3]['SBJ'].replace('NUM', '%.2d' % i )
        # compute variances
        fsl_var = compute_var(fsl_1, fsl_2, 'sbj{}-fsl-unthresh'.format('%.2d' % i), f3=fsl_3)
        afni_var = compute_var(afni_1, afni_2, 'sbj{}-afni-unthresh'.format('%.2d' % i), f3=afni_3)
        spm_var = compute_var(spm_1, spm_2, 'sbj{}-spm-unthresh'.format('%.2d' % i), f3=spm_3)
        # combine variances
        combine_var(fsl_var, afni_var, nib.load(fsl_1), 'sbj{}-fuzzy-fsl-afni-unthresh'.format('%.2d' % i))
        combine_var(fsl_var, spm_var, nib.load(fsl_1), 'sbj{}-fuzzy-fsl-spm-unthresh'.format('%.2d' % i))
        combine_var(afni_var, spm_var, nib.load(afni_1), 'sbj{}-fuzzy-afni-spm-unthresh'.format('%.2d' % i))


def get_ratio(path_):
    # Group-level
    for type_ in ['thresh', 'unthresh']:
        for pair_ in ['fsl-spm', 'fsl-afni', 'afni-spm']:
            img1_ = nib.load('{}{}-{}-std.nii.gz'.format(path_, pair_, type_))
            img2_ = nib.load('{}fuzzy-{}-{}-std.nii.gz'.format(path_, pair_, type_))
            img1_data = np.nan_to_num(img1_.get_fdata())
            img2_data = np.nan_to_num(img2_.get_fdata())
            ratio_ = img1_data / img2_data
            nft_img = nib.Nifti1Image(ratio_, img1_.affine, header=img1_.header)
            nib.save(nft_img, os.path.join(path_, 'ratio-{}-{}.nii.gz'.format(pair_, type_)))
    
    # Subject-level
    path_ = os.path.join(path_, 'subject_level')
    for i in range(1, 17):
        for type_ in ['unthresh']:
            for pair_ in ['fsl-spm', 'fsl-afni', 'afni-spm']:
                img1_ = nib.load('{}/sub-{}/sbj{}-{}-{}-std.nii.gz'.format(path_, '%.2d' % i, '%.2d' % i, pair_, type_))
                img2_ = nib.load('{}/sub-{}/sbj{}-fuzzy-{}-{}-std.nii.gz'.format(path_, '%.2d' % i, '%.2d' % i, pair_, type_))
                img1_data = np.nan_to_num(img1_.get_fdata())
                img2_data = np.nan_to_num(img2_.get_fdata())
                ratio_ = img1_data / img2_data
                nft_img = nib.Nifti1Image(ratio_, img1_.affine, header=img1_.header)
                nib.save(nft_img, os.path.join(path_, 'sbj{}-ratio-{}-{}.nii.gz'.format('%.2d' % i, pair_, type_)))


def get_diff(path_):
    # Group-level
    for type_ in ['thresh', 'unthresh']:
        for pair_ in ['fsl-spm', 'fsl-afni', 'afni-spm']:
            img1_ = nib.load('{}{}-{}-std.nii.gz'.format(path_, pair_, type_))
            img2_ = nib.load('{}fuzzy-{}-{}-std.nii.gz'.format(path_, pair_, type_))
            img1_data = np.nan_to_num(img1_.get_fdata())
            img2_data = np.nan_to_num(img2_.get_fdata())
            diff_ = img1_data - img2_data
            nft_img = nib.Nifti1Image(diff_, img1_.affine, header=img1_.header)
            nib.save(nft_img, os.path.join(path_, 'btMwt-{}-{}.nii.gz'.format(pair_, type_)))
    
    # Subject-level
    path_ = os.path.join(path_, 'subject_level')
    for i in range(1, 17):
        for type_ in ['unthresh']:
            for pair_ in ['fsl-spm', 'fsl-afni', 'afni-spm']:
                img1_ = nib.load('{}/sub-{}/sbj{}-{}-{}-std.nii.gz'.format(path_, '%.2d' % i, '%.2d' % i, pair_, type_))
                img2_ = nib.load('{}/sub-{}/sbj{}-fuzzy-{}-{}-std.nii.gz'.format(path_, '%.2d' % i, '%.2d' % i, pair_, type_))
                img1_data = np.nan_to_num(img1_.get_fdata())
                img2_data = np.nan_to_num(img2_.get_fdata())
                diff_ = img1_data - img2_data
                nft_img = nib.Nifti1Image(diff_, img1_.affine, header=img1_.header)
                nib.save(nft_img, os.path.join(path_, 'sub-{}/sbj{}-btMwt-{}-{}.nii.gz'.format('%.2d' % i, '%.2d' % i, pair_, type_)))


def cluster_std(bt_, wt_, img_f1, tool_, type_):
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
    nib.save(img_d_img, './data/std/clusters/upper-maps-{}-{}.nii.gz'.format(type_, tool_))

    # Lower maps
    bt_class2 = np.where((wt_<0.1), bt_, np.nan)#
    bt_data2 = np.reshape(bt_class2, -1)
    wt_class2 = np.where((wt_<0.1), wt_, np.nan)#
    wt_data2 = np.reshape(wt_class2, -1)
    lower_map = np.where((wt_<0.1), O, np.nan)#
    img_d_img = nib.Nifti1Image(lower_map, img_f1.affine, header=img_f1.header)
    nib.save(img_d_img, './data/std/clusters/lower-maps-{}-{}.nii.gz'.format(type_, tool_))

    # Correlated maps
    # (0.5<bt_/wt_) & (bt_/wt_< 2) & (wt_ > 0.1) & (bt_ > 0.1)
    bt_class3 = np.where((0.5<bt_/wt_) & (bt_/wt_< 2) & (wt_ > 0.1) & (bt_ > 0.1), bt_, np.nan)#
    bt_data3 = np.reshape(bt_class3, -1)
    wt_class3 = np.where((0.5<bt_/wt_) & (bt_/wt_< 2) & (wt_ > 0.1) & (bt_ > 0.1), wt_, np.nan)#
    wt_data3 = np.reshape(wt_class3, -1)
    corr_map = np.where((0.5<bt_/wt_) & (bt_/wt_< 2) & (wt_ > 0.1) & (bt_ > 0.1), bt_/wt_, np.nan)#
    img_d_img = nib.Nifti1Image(corr_map, img_f1.affine, header=img_f1.header)
    nib.save(img_d_img, './data/std/clusters/correlated-maps-{}-{}.nii.gz'.format(type_, tool_))

    return bt_data1, wt_data1, bt_data2, wt_data2, bt_data3, wt_data3, corr_map, upper_map


def plot_corr_variances():
    ### Plot correlation of variances between BT and WT
    for ind1, type_ in enumerate(['unthresh']):
        fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(24, 7))
        #fig.suptitle("Correlation of variances between tool-variability vs. numerical-variability")
        for ind_, pair_ in enumerate(['fsl-spm', 'fsl-afni', 'afni-spm']):
            bfuzzy = './data/std/fuzzy-{}-{}-std.nii.gz'.format(pair_, type_)
            bfuzzy = nib.load(bfuzzy)
            btool = './data/std/{}-{}-std.nii.gz'.format(pair_, type_)
            btool = nib.load(btool)
            if btool.shape != bfuzzy.shape:
                raise NameError('Images from BT and WT are from different dimensions!')

            wt_ = bfuzzy.get_fdata()
            bt_ = btool.get_fdata()
            data1 = np.reshape(bt_, -1)
            data1 = np.nan_to_num(data1)
            data2 = np.reshape(wt_, -1)
            data2 = np.nan_to_num(data2)

            slope, intercept, r, p, stderr = scipy.stats.linregress(data1, data2)
            #line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'
            y = intercept + slope * data1

            t1, t2 = pair_.split('-')
            label_ = "{} and {}".format(t1.upper(), t2.upper()) #'{}'.format(pair_.upper())
            # if single: label_ = '{}'.format(pair_.upper())

            bt_data1, wt_data1, bt_data2, wt_data2, bt_data3, wt_data3, corr_map, upper_map = cluster_std(bt_, wt_, btool, pair_, type_)
            ax[ind_].plot(data1, data2, linewidth=0, marker='o', alpha=.5, label=label_)
            #ax[ind_].plot(bt_data1, wt_data1, linewidth=0, marker='o', color='purple', alpha=.5, label='Upper cluster')
            #ax[ind_].plot(bt_data2, wt_data2, linewidth=0, marker='o', color='yellow', alpha=.5, label='Lower cluster')
            #ax[ind_].plot(bt_data3, wt_data3, linewidth=0, marker='o', color='green', alpha=.5, label='Identity line')

            max_bt = max(np.nan_to_num(bt_data3))
            bt_max_ind = np.argmax(np.nan_to_num(bt_data3))
            wt_lower = wt_data3[bt_max_ind]

            ax[ind_].plot([0, 2.5], [0, 2.5], color='black', linestyle='dashed', label='Identity line')

            # ax[ind_].plot([0., 3.7], [0., 1.85], color='black', linestyle='dashed', label='Identity line')
            # ax[ind_].plot([0., 1.25], [0., 2.5], color='black', linestyle='dashed')

            # ci = 1.96 * np.std(y)/np.mean(y)
            # ax.fill_between(x, (y-ci), (y+ci), color='g', alpha=.9)
            ax[ind_].set_title('')
            ax[ind_].set_xlabel('')
            if ind_ == 0: ax[ind_].set_ylabel('WT variability', fontsize=14)
            # if ind_[0] == 0: ax[ind_[0]].set_title('Correlation of variances in {}olded maps'.format(type_))
            ax[ind_].set_xlabel('BT variability', fontsize=14)
            ax[ind_].set_xlim([-0.15, 5])
            ax[ind_].set_ylim([-0.07, 2.6])
            # ax[ind_].set_xticklabels(fontsize=14)
            # ax[ind_].set_yticklabels(fontsize=14)
            ax[ind_].legend(facecolor='white', loc='upper right', fontsize=12)
            ax[ind_].tick_params(axis='both', labelsize=12)

            #ax[ind_].set_xscale('log')

            # print fraction of clusters
            data2[(data2 == 0) & (data1 == 0)] = np.nan
            all_corr_count = np.count_nonzero(~np.isnan(data2))
            corr_map[np.isnan(bt_) & np.isnan(wt_)] = np.nan
            corr_count = np.count_nonzero(~np.isnan(corr_map))
            upper_map[np.isnan(bt_) & np.isnan(wt_)] = np.nan
            upper_count = np.count_nonzero(~np.isnan(upper_map))
            print('{}\nFraction of green: %{:.1f}\nFraction of purple: %{:.1f}'.format(pair_, (corr_count/all_corr_count)*100, (upper_count/all_corr_count)*100))
            print('stop')

        #plt.show()
        plt.savefig('./paper/figures/std-corr-{}-plot.png'.format(type_), bbox_inches='tight')


############# COMPUTE and PLOT DICES


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

            dices_[regions[r]][act_]['tool']['fsl-afni'] = compute_dice(masked_regions['fsl'], masked_regions['afni'])[0]
            dices_[regions[r]][act_]['tool']['fsl-spm'] = compute_dice(masked_regions['fsl'], masked_regions['spm'])[0]
            dices_[regions[r]][act_]['tool']['afni-spm'] = compute_dice(masked_regions['afni'], masked_regions['spm'])[0]
            # dices = (dice_res_1, dark_dice_1[1], dark_dice_2[1], num_activated_1, num_activated_2)

            for tool_ in mca_results.keys():
                masked_regions['{}1'.format(tool_)] = keep_roi(mca_results[tool_][1][act_], r, image_parc)#, '{}1'.format(tool_))
                masked_regions['{}2'.format(tool_)] = keep_roi(mca_results[tool_][2][act_], r, image_parc)#, '{}2'.format(tool_))
                masked_regions['{}3'.format(tool_)] = keep_roi(mca_results[tool_][3][act_], r, image_parc)#, '{}3'.format(tool_))

                dices_[regions[r]][act_]['mca']['{}1'.format(tool_)] = compute_dice(masked_regions['{}1'.format(tool_)], masked_regions['{}2'.format(tool_)])[0]
                dices_[regions[r]][act_]['mca']['{}2'.format(tool_)] = compute_dice(masked_regions['{}1'.format(tool_)], masked_regions['{}3'.format(tool_)])[0]
                dices_[regions[r]][act_]['mca']['{}3'.format(tool_)] = compute_dice(masked_regions['{}2'.format(tool_)], masked_regions['{}3'.format(tool_)])[0]

    dump_variable(dices_, retrieve_name(dices_)[0])
    return dices_


def plot_dices(dices_):
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(16, 12))
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
            line_norm = f'Regression line: y={intercept_norm:.2f}+{slope_norm:.2f}x' #, r={r_norm:.2f}, p={p_norm:.10f}
            y_norm = intercept_norm + slope_norm * np.array(tool_list)

            ax.plot(tool_norm, mca_norm, linewidth=0, alpha=.5, color=colors[ind_], marker='o', label='{}'.format(tool_.upper()))
            ax.plot(tool_list, y_norm, color=colors[ind_], alpha=.7, label=line_norm)
            ax.set_xlabel('Normalized Dice scores in BT', fontsize=22)
            ax.set_ylabel('Normalized Dice scores in WT', fontsize=22)
            # ax.set_title('Normalized Dice scores from thresholded maps')
            ax.legend(fontsize=16)
            ax.tick_params(axis='both', labelsize=18)
    #plt.show()
    plt.savefig('./paper/figures/dices_corr.png', bbox_inches='tight')


############# PRINT STATS

def print_gl_stats(path_):
    # Compute stats of variabilities (ignore NaNs as bg)    
    for type_ in ['thresh', 'unthresh']:
        for pair_ in ['fsl-spm', 'fsl-afni', 'afni-spm']:
            bt_ = nib.load('{}{}-{}-std.nii.gz'.format(path_, pair_, type_))
            bt_std_data = bt_.get_fdata()
            bt_std_mean = np.nanmean(bt_std_data)
            bt_std_std = np.nanstd(bt_std_data)
            print('BT variability of tstats in {} {}:\nMean {}\nStd. {}'.format(pair_, type_, bt_std_mean, bt_std_std))

        for tool_ in ['fsl', 'spm', 'afni']:
            wt2_ = nib.load('{}{}-{}-std.nii.gz'.format(path_, tool_, type_))
            wt2_std_data = wt2_.get_fdata()
            wt2_std_mean = np.nanmean(wt2_std_data)
            wt2_std_std = np.nanstd(wt2_std_data)
            print('WT variability of tstats in {} {}:\nMean {}\nStd. {}'.format(tool_, type_, wt2_std_mean, wt2_std_std))

        print("stop")


def print_sl_stats(path_):
    # Compute Mean of std over subjects (ignore NaNs as bg)
    min_sbj = 100
    max_sbj = 0
    for i in range(1, 17):
        wt_list = []
        bt_list = []
        for tool_ in ['fsl', 'spm', 'afni']:
            wt_ = nib.load(os.path.join(path_, 'sbj{}-{}-unthresh-std.nii.gz'.format('%.2d' % i, tool_)))
            wt_std_data = wt_.get_fdata()
            wt_std_mean = np.nanmean(wt_std_data)
            wt_list.append(wt_std_mean)

        for pair_ in ['fsl-spm', 'fsl-afni', 'afni-spm']:
            bt_ = nib.load(os.path.join(path_, 'sbj{}-{}-unthresh-std.nii.gz'.format('%.2d' % i, pair_)))
            bt_std_data = bt_.get_fdata()
            bt_std_mean = np.nanmean(bt_std_data)
            bt_list.append(bt_std_mean)

        if mean(wt_list) > max_sbj:
            max_sbj = mean(wt_list)
            i_max = i
            max_sbj_bt = mean(bt_list)

    print('Subject {} has the highest WT variability as the average std of {}\n BT std is {}'.format(i_max, max_sbj, max_sbj_bt))
    print('stop')


def compute_stat_test():
    path_ = 'data/std/'
    for type_ in ['unthresh', 'thresh']:
        for pair_ in ['fsl-spm', 'fsl-afni', 'afni-spm']:
            bt_ = nib.load('{}{}-{}-std.nii.gz'.format(path_, pair_, type_))
            bt_std_data = bt_.get_fdata()

            t1, t2 = pair_.split('-')
            wt1_ = nib.load('{}{}-{}-std.nii.gz'.format(path_, t1, type_))
            wt1_std_data = wt1_.get_fdata()

            wt2_ = nib.load('{}{}-{}-std.nii.gz'.format(path_, t2, type_))
            wt2_std_data = wt2_.get_fdata()

            bg_ = np.where((np.isnan(bt_.get_fdata())) , False, True)
            bt_img = np.nan_to_num(bt_std_data)[bg_]

            bg_ = np.where((np.isnan(wt1_.get_fdata())) , False, True)
            img1 = np.nan_to_num(wt1_std_data)[bg_]
            bg_ = np.where((np.isnan(wt2_.get_fdata())) , False, True)
            img2 = np.nan_to_num(wt2_std_data)[bg_]
            # print('WT img1 mean {} and img2 mean {} and bt_img mean {} '.format(mean(img1), mean(img2), mean(bt_img)))
            num_sample = int(1e3)
            res_bt = resample(bt_img, num_sample)
            res1_ = resample(img1, num_sample)
            res2_ = resample(img2, num_sample)

            t_stat1, p_val1 = stat.wilcoxon(res_bt, res1_)
            t_stat2, p_val2 = stat.wilcoxon(res_bt, res2_)
            print("wilcoxon-test between BT({}) and WT({}) is {} and p-value {}".format(pair_, t1, t_stat1, p_val1*6))
            print("wilcoxon-test between BT({}) and WT({}) is {} and p-value {}".format(pair_, t2, t_stat2, p_val2*6))

            t_stat1, p_val1 = stat.ttest_ind(res_bt, res1_)
            t_stat2, p_val2 = stat.ttest_ind(res_bt, res2_)

            print("t-test between BT({}) and WT({}) is {} and p-value {}".format(pair_, t1, t_stat1, p_val1*6))
            print("t-test between BT({}) and WT({}) is {} and p-value {}".format(pair_, t2, t_stat2, p_val2*6))
            print('stop')


def combine_thresh(tool_results, mca_results):
    ## Combine activation and deactivation of thresholded maps ###
    # between tool
    for i in tool_results.keys():
        path_ = os.path.dirname(tool_results[i]['exc_set_file'])
        n = nib.load(tool_results[i]['exc_set_file'])
        d = n.get_data()
        exc_set_nonan = nib.Nifti1Image(np.nan_to_num(d), n.affine, header=n.header)
        n = nib.load(tool_results[i]['exc_set_file_neg'])
        d = n.get_data()

        exc_set_neg_nonan = nib.Nifti1Image(np.nan_to_num(d), n.affine, header=n.header)
        to_display = math_img("img1-img2", img1=exc_set_nonan, img2=exc_set_neg_nonan)
        nib.save(to_display, os.path.join(path_, '{}.nii.gz'.format(i)))

    # within tool
    for tool_ in mca_results.keys():
        dic_ = mca_results[tool_]
        for i in dic_.keys():
            path_ = os.path.dirname(dic_[i]['exc_set_file'])
            n = nib.load(dic_[i]['exc_set_file'])
            d = n.get_data()
            exc_set_nonan = nib.Nifti1Image(np.nan_to_num(d), n.affine, header=n.header)
            n = nib.load(dic_[i]['exc_set_file_neg'])
            d = n.get_data()

            exc_set_neg_nonan = nib.Nifti1Image(np.nan_to_num(d), n.affine, header=n.header)
            to_display = math_img("img1-img2", img1=exc_set_nonan, img2=exc_set_neg_nonan)
            nib.save(to_display, os.path.join(path_, '{}_s{}.nii.gz'.format(tool_, i)))


def main(args=None):

    tool_results = {}
    tool_results['fsl'] = {}
    tool_results['fsl']['exc_set_file'] = './results/ds000001/tools/FSL/thresh_zstat1.nii.gz'
    tool_results['fsl']['exc_set_file_neg'] = './results/ds000001/tools/FSL/thresh_zstat2.nii.gz'
    tool_results['fsl']['stat_file'] = './results/ds000001/tools/FSL/tstat1.nii.gz'
    tool_results['fsl']['act_deact'] = './results/ds000001/tools/FSL/fsl.nii.gz'
    tool_results['fsl']['SBJ'] = './results/ds000001/tools/FSL/subject_level/sbjNUM_tstat1.nii.gz' # NUM will replace with the sbj number
    tool_results['spm'] = {}
    tool_results['spm']['exc_set_file'] = './results/ds000001/tools/SPM/Octave/spm_exc_set.nii.gz'
    tool_results['spm']['exc_set_file_neg'] = './results/ds000001/tools/SPM/Octave/spm_exc_set_neg.nii.gz'
    tool_results['spm']['stat_file'] = './results/ds000001/tools/SPM/Octave/spm_stat.nii.gz'
    # tool_results['spm']['exc_set_file'] = './results/ds000001/tools/SPM/Matlab2016b/spm_exc_set.nii.gz'
    # tool_results['spm']['exc_set_file_neg'] = './results/ds000001/tools/SPM/Matlab2016b/spm_exc_set_neg.nii.gz'
    # tool_results['spm']['stat_file'] = './results/ds000001/tools/SPM/Matlab2016b/spm_stat.nii.gz'
    tool_results['spm']['act_deact'] = './results/ds000001/tools/SPM/Octave/spm.nii.gz'
    tool_results['spm']['SBJ'] = './results/ds000001/tools/SPM/Octave/subject_level/sub-NUM/spm_stat.nii.gz' # NUM will replace with the sbj number
    tool_results['afni'] = {}
    tool_results['afni']['exc_set_file'] = './results/ds000001/tools/AFNI/Positive_clustered_t_stat.nii.gz'
    tool_results['afni']['exc_set_file_neg'] = './results/ds000001/tools/AFNI/Negative_clustered_t_stat.nii.gz'
    tool_results['afni']['stat_file'] = './results/ds000001/tools/AFNI/3dMEMA_result_t_stat_masked.nii.gz'
    tool_results['afni']['act_deact'] = './results/ds000001/tools/AFNI/afni.nii.gz'
    tool_results['afni']['SBJ'] = './results/ds000001/tools/AFNI/subject_level/tstats/sbjNUM_result_t_stat_masked.nii.gz' # NUM will replace with the sbj number

    fsl_mca = {}
    fsl_mca[1] = {}
    fsl_mca[1]['exc_set_file'] = "./results/ds000001/fuzzy/p53/FSL/run1/thresh_zstat1_53_run1.nii.gz"
    fsl_mca[1]['exc_set_file_neg'] = "./results/ds000001/fuzzy/p53/FSL/run1/thresh_zstat2_53_run1.nii.gz"
    fsl_mca[1]['stat_file'] = "./results/ds000001/fuzzy/p53/FSL/run1/tstat1_53_run1.nii.gz"
    fsl_mca[1]['act_deact'] = "./results/ds000001/fuzzy/p53/FSL/run1/fsl_s1.nii.gz"
    fsl_mca[1]['SBJ'] = './results/ds000001/fuzzy/p53/FSL/subject_level/run1/sbjNUM_tstat1.nii.gz'
    fsl_mca[2] = {}
    fsl_mca[2]['exc_set_file'] = "./results/ds000001/fuzzy/p53/FSL/run2/thresh_zstat1_53_run2.nii.gz"
    fsl_mca[2]['exc_set_file_neg'] = "./results/ds000001/fuzzy/p53/FSL/run2/thresh_zstat2_53_run2.nii.gz"
    fsl_mca[2]['stat_file'] = "./results/ds000001/fuzzy/p53/FSL/run2/tstat1_53_run2.nii.gz"
    fsl_mca[2]['act_deact'] = "./results/ds000001/fuzzy/p53/FSL/run2/fsl_s2.nii.gz"
    fsl_mca[2]['SBJ'] = './results/ds000001/fuzzy/p53/FSL/subject_level/run2/sbjNUM_tstat1.nii.gz'
    fsl_mca[3] = {}
    fsl_mca[3]['exc_set_file'] = "./results/ds000001/fuzzy/p53/FSL/run3/thresh_zstat1_53_run3.nii.gz"
    fsl_mca[3]['exc_set_file_neg'] = "./results/ds000001/fuzzy/p53/FSL/run3/thresh_zstat2_53_run3.nii.gz"
    fsl_mca[3]['stat_file'] = "./results/ds000001/fuzzy/p53/FSL/run3/tstat1_53_run3.nii.gz"
    fsl_mca[3]['act_deact'] = "./results/ds000001/fuzzy/p53/FSL/run3/fsl_s3.nii.gz"
    fsl_mca[3]['SBJ'] = './results/ds000001/fuzzy/p53/FSL/subject_level/run3/sbjNUM_tstat1.nii.gz'

    spm_mca = {}
    spm_mca[1] = {}
    spm_mca[1]['exc_set_file'] = "./results/ds000001/fuzzy/p53/SPM-Octace/run1/spm_exc_set.nii.gz"
    spm_mca[1]['exc_set_file_neg'] = "./results/ds000001/fuzzy/p53/SPM-Octace/run1/spm_exc_set_neg.nii.gz"
    spm_mca[1]['stat_file'] = "./results/ds000001/fuzzy/p53/SPM-Octace/run1/spm_stat.nii.gz"
    spm_mca[1]['act_deact'] = "./results/ds000001/fuzzy/p53/SPM-Octace/run1/spm_s1.nii.gz"
    spm_mca[1]['SBJ'] = './results/ds000001/fuzzy/p53/SPM-Octace/subject_level/run1/sub-NUM/spm_stat.nii.gz'
    spm_mca[2] = {}
    spm_mca[2]['exc_set_file'] = "./results/ds000001/fuzzy/p53/SPM-Octace/run2/spm_exc_set.nii.gz"
    spm_mca[2]['exc_set_file_neg'] = "./results/ds000001/fuzzy/p53/SPM-Octace/run2/spm_exc_set_neg.nii.gz"
    spm_mca[2]['stat_file'] = "./results/ds000001/fuzzy/p53/SPM-Octace/run2/spm_stat.nii.gz"
    spm_mca[2]['act_deact'] = "./results/ds000001/fuzzy/p53/SPM-Octace/run2/spm_s2.nii.gz"
    spm_mca[2]['SBJ'] = './results/ds000001/fuzzy/p53/SPM-Octace/subject_level/run2/sub-NUM/spm_stat.nii.gz'
    spm_mca[3] = {}
    spm_mca[3]['exc_set_file'] = "./results/ds000001/fuzzy/p53/SPM-Octace/run3/spm_exc_set.nii.gz"
    spm_mca[3]['exc_set_file_neg'] = "./results/ds000001/fuzzy/p53/SPM-Octace/run3/spm_exc_set_neg.nii.gz"
    spm_mca[3]['stat_file'] = "./results/ds000001/fuzzy/p53/SPM-Octace/run3/spm_stat.nii.gz"
    spm_mca[3]['act_deact'] = "./results/ds000001/fuzzy/p53/SPM-Octace/run3/spm_s3.nii.gz"
    spm_mca[3]['SBJ'] = './results/ds000001/fuzzy/p53/SPM-Octace/subject_level/run3/sub-NUM/spm_stat.nii.gz'

    afni_mca = {}
    afni_mca[1] = {}
    afni_mca[1]['exc_set_file'] = "./results/ds000001/fuzzy/p53/AFNI/run1/Positive_clustered_t_stat.nii.gz"
    afni_mca[1]['exc_set_file_neg'] = "./results/ds000001/fuzzy/p53/AFNI/run1/Negative_clustered_t_stat.nii.gz"
    afni_mca[1]['stat_file'] = "./results/ds000001/fuzzy/p53/AFNI/run1/3dMEMA_result_t_stat_masked.nii.gz"
    afni_mca[1]['act_deact'] = "./results/ds000001/fuzzy/p53/AFNI/run1/afni_s1.nii.gz"
    afni_mca[1]['SBJ'] = './results/ds000001/fuzzy/p53/AFNI/subject_level/tstats-run1/sbjNUM_result_t_stat_masked.nii.gz'
    afni_mca[2] = {}
    afni_mca[2]['exc_set_file'] = "./results/ds000001/fuzzy/p53/AFNI/run2/Positive_clustered_t_stat.nii.gz"
    afni_mca[2]['exc_set_file_neg'] = "./results/ds000001/fuzzy/p53/AFNI/run2/Negative_clustered_t_stat.nii.gz"
    afni_mca[2]['stat_file'] = "./results/ds000001/fuzzy/p53/AFNI/run2/3dMEMA_result_t_stat_masked.nii.gz"
    afni_mca[2]['act_deact'] = "./results/ds000001/fuzzy/p53/AFNI/run2/afni_s2.nii.gz"
    afni_mca[2]['SBJ'] = './results/ds000001/fuzzy/p53/AFNI/subject_level/tstats-run2/sbjNUM_result_t_stat_masked.nii.gz'
    afni_mca[3] = {}
    afni_mca[3]['exc_set_file'] = "./results/ds000001/fuzzy/p53/AFNI/run3/Positive_clustered_t_stat.nii.gz"
    afni_mca[3]['exc_set_file_neg'] = "./results/ds000001/fuzzy/p53/AFNI/run3/Negative_clustered_t_stat.nii.gz"
    afni_mca[3]['stat_file'] = "./results/ds000001/fuzzy/p53/AFNI/run3/3dMEMA_result_t_stat_masked.nii.gz"
    afni_mca[3]['act_deact'] = "./results/ds000001/fuzzy/p53/AFNI/run3//afni_s3.nii.gz"
    afni_mca[3]['SBJ'] = './results/ds000001/fuzzy/p53/AFNI/subject_level/tstats-run3/sbjNUM_result_t_stat_masked.nii.gz'

    mca_results = {}
    mca_results['fsl'] = {}
    mca_results['fsl'] = fsl_mca
    mca_results['spm'] = {}
    mca_results['spm'] = spm_mca
    mca_results['afni'] = {}
    mca_results['afni'] = afni_mca
    
    std_path = 'data/std/'
    ### Combine activation and deactivation maps
    # combine_thresh(tool_results, mca_results)

    ### Create std images
    # var_between_tool(tool_results) #BT
    # var_between_fuzzy(mca_results) #WT
    ### Create ratio images between BT and WT std images
    # get_ratio(std_path)
    # get_diff(std_path)
    ### Plot correlation of variances between BT and FL (Fig 4)
    #plot_corr_variances()
    ### std in different precisions in WT
    # path_ = './results/ds000001/fuzzy/'
    # compute_std_WT(path_)
    ### Global nearest precision
    # p_nearest, all_rmse = global_nearest_precision()
    # print(p_nearest)
    # plot_rmse_nearest(all_rmse)

    ### Compute Dice scores and then plot (Fig 2)
    # image_parc = './data/MNI-parcellation/HCPMMP1_on_MNI152_ICBM2009a_nlin_resampled.splitLR.nii.gz'
    # regions_txt = './data/MNI-parcellation/HCP-MMP1_on_MNI152_ICBM2009a_nlin.txt'
    # if os.path.exists('./data/dices_.pkl'):
    #     dices_ = load_variable('dices_')
    #     plot_dices(dices_)
    # else:
    #     dices_ = get_dice_values(regions_txt, image_parc, tool_results, mca_results)
    #     plot_dices(dices_)

    ### Print stats (Table 2)
    # print_gl_stats(std_path)
    # print_sl_stats(std_path)
    # compute_stat_test()

if __name__ == '__main__':
    main()