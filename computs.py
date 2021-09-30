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
import os


############# COMPUTES STD. DEV.

def compute_var(f1, f2, file_name=None, f3=None):
    # Compute std. between tool image samples
    f1 = nib.load(f1)
    img_concat = nib.funcs.concat_images([f1, f2], check_affines=True, axis=None)
    img_std = np.std(img_concat.get_fdata(), axis=3)

    # Compute variance between fuzzy image samples
    if f3 is not None:
        img_concat = nib.funcs.concat_images([f1, f2, f3], check_affines=True, axis=None)
        img_var = np.var(img_concat.get_fdata(), axis=3)
        nft_img = nib.Nifti1Image(img_var, f1.affine, header=f1.header)
        return nft_img

    # nan bg images
    img_std[img_std == 0] = np.nan
    # mean_ = np.nanmean(img_std)
    # print('mean of {} is {}'.format(file_name, mean_))

    nft_img = nib.Nifti1Image(img_std, f1.affine, header=f1.header)
    nib.save(nft_img, os.path.join('data/std/', '{}-std.nii.gz'.format(file_name)))


def combine_var(f1, f2, meta_, file_name):
    var_f2_res = resample_from_to(f2, f1, order=0)
    # to combine two image variances, we use: var(x+y) = var(x) + var(y) + 2*cov(x,y)
    # and since the correlation between two arrays are so weak, we droped `2*cov(x,y)` from the formula
    combine_var = np.nan_to_num(f1.get_fdata()) + np.nan_to_num(var_f2_res.get_fdata())
    std_ = np.sqrt(combine_var)
    # nan bg images
    std_[std_ == 0] = np.nan
    # mean_ = np.nanmean(std_)
    # print('mean of {} is {}'.format(file_name, mean_))

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
    fsl_var = compute_var(fsl_1, fsl_2, f3=fsl_3)
    afni_var = compute_var(afni_1, afni_2, f3=afni_3)
    spm_var = compute_var(spm_1, spm_2, f3=spm_3)
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
    fsl_var = compute_var(fsl_1, fsl_2, f3=fsl_3)
    afni_var = compute_var(afni_1, afni_2, f3=afni_3)
    spm_var = compute_var(spm_1, spm_2, f3=spm_3)
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
        fsl_var = compute_var(fsl_1, fsl_2, f3=fsl_3)
        afni_var = compute_var(afni_1, afni_2, f3=afni_3)
        spm_var = compute_var(spm_1, spm_2, f3=spm_3)
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
                img1_ = nib.load('{}/sbj{}-{}-{}-std.nii.gz'.format(path_, '%.2d' % i, pair_, type_))
                img2_ = nib.load('{}/sbj{}-fuzzy-{}-{}-std.nii.gz'.format(path_, '%.2d' % i, pair_, type_))
                img1_data = np.nan_to_num(img1_.get_fdata())
                img2_data = np.nan_to_num(img2_.get_fdata())
                ratio_ = img1_data / img2_data
                nft_img = nib.Nifti1Image(ratio_, img1_.affine, header=img1_.header)
                nib.save(nft_img, os.path.join(path_, 'sbj{}-ratio-{}-{}.nii.gz'.format('%.2d' % i, pair_, type_)))


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
            ax[ind_].plot(bt_data1, wt_data1, linewidth=0, marker='o', color='purple', alpha=.5, label='Upper cluster')
            #ax[ind_].plot(bt_data2, wt_data2, linewidth=0, marker='o', color='yellow', alpha=.5, label='Lower cluster')
            ax[ind_].plot(bt_data3, wt_data3, linewidth=0, marker='o', color='green', alpha=.5, label='Correlated cluster')

            #ax[ind_].plot(data1, y, color='black', alpha=.5, label='Regression line')
            # ci = 1.96 * np.std(y)/np.mean(y)
            # ax.fill_between(x, (y-ci), (y+ci), color='g', alpha=.9)
            ax[ind_].set_title('')
            ax[ind_].set_xlabel('')
            if ind_ == 0: ax[ind_].set_ylabel('Within tool variation (std.)')
            # if ind_[0] == 0: ax[ind_[0]].set_title('Correlation of variances in {}olded maps'.format(type_))
            ax[ind_].set_xlabel('Between tool variation (std.)')
            ax[ind_].set_xlim([-0.15, 5])
            ax[ind_].set_ylim([-0.07, 2.6])
            ax[ind_].legend(facecolor='white', loc='upper right')
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


############# COMPUTES DICES


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
            line_norm = f'Regression line: y={intercept_norm:.2f}+{slope_norm:.2f}x' #, r={r_norm:.2f}, p={p_norm:.10f}
            y_norm = intercept_norm + slope_norm * np.array(tool_list)

            ax.plot(tool_norm, mca_norm, linewidth=0, alpha=.5, color=colors[ind_], marker='o', label='{}'.format(tool_.upper()))
            ax.plot(tool_list, y_norm, color=colors[ind_], alpha=.7, label=line_norm)
            ax.set_xlabel('Normalized Dice scores in BT')
            ax.set_ylabel('Normalized Dice scores in WT')
            # ax.set_title('Normalized Dice scores from thresholded maps')
            ax.legend()
    #plt.show()
    plt.savefig('./paper/figures/dices_corr.png', bbox_inches='tight')


############# PRINTS STATS

def compute_mean(data1_file, data2_file):
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


def print_gl_stats(tool_results, mca_results):
    # Compute Mean of diff over whole image (no mask for bg)
    for type_ in ['stat_file', 'act_deact']: # 'stat_file': 'unthresh' and 'act_deact': 'thresh'
        for pair_ in ['fsl-spm', 'fsl-afni', 'afni-spm']:
            t1, t2 = pair_.split('-')
            f1 = tool_results[t1][type_]
            f2 = tool_results[t2][type_]
            bt_mean_diff = compute_mean(f1, f2)

            # Fuzzy samples
            means_ = []
            means_.append(compute_mean(mca_results[t1][1][type_], mca_results[t1][2][type_]))
            means_.append(compute_mean(mca_results[t1][1][type_], mca_results[t1][3][type_]))
            means_.append(compute_mean(mca_results[t1][2][type_], mca_results[t1][3][type_]))
            means_.append(compute_mean(mca_results[t2][1][type_], mca_results[t2][2][type_]))
            means_.append(compute_mean(mca_results[t2][1][type_], mca_results[t2][3][type_]))
            means_.append(compute_mean(mca_results[t2][2][type_], mca_results[t2][3][type_]))
            wt_mean_diff = mean(means_)
            print('Mean of diff of tstats in {} {}:\nBT {}\nWT {}'.format(pair_, type_, bt_mean_diff, wt_mean_diff))

    # Compute Std of differences (ignore NaNs/zeros as bg)    
    path_ = './data/std/'
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
            print('Std of diff of tstats in {} {}:\nBT {}\nWT {}'.format(pair_, type_, bt_std_mean, wt_std_mean))
            # # Compute SE: n=2 for bt and n=6 for wt
            # bt_se_data = bt_std_data/np.sqrt(2)
            # wt_se_data = wt_std_data/np.sqrt(6)
            # bt_se_mean = np.nanmean(bt_se_data)
            # wt_se_mean = np.nanmean(wt_se_data)
            # print('SE of diff of tstats in {} {}:\nBT {}\nWT {}'.format(pair_, type_, bt_se_mean, wt_se_mean))
            print("stop")


def print_sl_stats(tool_results, mca_results):
    # Compute Mean of diff over whole image (no mask for bg)
    min_sbj = 100
    max_sbj = 0
    for i in range(1, 17): 
        bt_mean = []
        wt_mean = []
        for pair_ in ['fsl-spm', 'fsl-afni', 'afni-spm']:
            t1, t2 = pair_.split('-')
            f1 = tool_results[t1]['SBJ'].replace('NUM', '%.2d' % i )
            f2 = tool_results[t2]['SBJ'].replace('NUM', '%.2d' % i )
            bt_mean_diff = compute_mean(f1, f2)
            bt_mean.append(bt_mean_diff)

            # Fuzzy samples
            means_ = []
            means_.append(compute_mean(mca_results[t1][1]['SBJ'].replace('NUM', '%.2d' % i ),
                                       mca_results[t1][2]['SBJ'].replace('NUM', '%.2d' % i )))
            means_.append(compute_mean(mca_results[t1][1]['SBJ'].replace('NUM', '%.2d' % i ),
                                       mca_results[t1][3]['SBJ'].replace('NUM', '%.2d' % i )))
            means_.append(compute_mean(mca_results[t1][2]['SBJ'].replace('NUM', '%.2d' % i ),
                                       mca_results[t1][3]['SBJ'].replace('NUM', '%.2d' % i )))
            means_.append(compute_mean(mca_results[t2][1]['SBJ'].replace('NUM', '%.2d' % i ),
                                       mca_results[t2][2]['SBJ'].replace('NUM', '%.2d' % i )))
            means_.append(compute_mean(mca_results[t2][1]['SBJ'].replace('NUM', '%.2d' % i ),
                                       mca_results[t2][3]['SBJ'].replace('NUM', '%.2d' % i )))
            means_.append(compute_mean(mca_results[t2][2]['SBJ'].replace('NUM', '%.2d' % i ),
                                       mca_results[t2][3]['SBJ'].replace('NUM', '%.2d' % i )))
            wt_mean_diff = mean(means_)
            wt_mean.append(wt_mean_diff)
            # print('Mean of diff of unthresholded tstats in Subject {} \n{}:\nBT {}\nWT {}'.format('%.2d' % i, pair_, bt_mean_diff, wt_mean_diff))
            # print('Stop')
        avg_bt_wt = (mean(wt_mean) + mean(bt_mean))/2
        if avg_bt_wt < min_sbj:
            min_sbj = avg_bt_wt
            i_min = i
        if avg_bt_wt > max_sbj:
            max_sbj = avg_bt_wt
            i_max = i
        # print('{}: bt_mean {} and wt_mean {}'.format(pair_, mean(bt_mean), mean(wt_mean)))
    print('i_min {} mean {} \ni_max {} mean {}'.format(i_min, min_sbj, i_max, max_sbj))
    print('stop')

    # Compute Std of differences (ignore NaNs/zeros as bg)    
    path_ = './data/std/subject_level'
    type_ = 'unthresh'
    min_sbj = 100
    max_sbj = 0
    for i in range(1, 17):
        bt_mean = []
        wt_mean = []
        for pair_ in ['fsl-spm', 'fsl-afni', 'afni-spm']:
            img1_ = nib.load('{}/sub-{}/sbj{}-{}-{}-std.nii.gz'.format(path_, '%.2d' % i, '%.2d' % i, pair_, type_))
            img2_ = nib.load('{}/sub-{}/sbj{}-fuzzy-{}-{}-std.nii.gz'.format(path_, '%.2d' % i, '%.2d' % i, pair_, type_))
            # img1_data = np.nan_to_num(img1_.get_fdata())
            # img2_data = np.nan_to_num(img2_.get_fdata())
            bt_std_data = img1_.get_fdata()
            wt_std_data = img2_.get_fdata()
            bt_std_mean = np.nanmean(bt_std_data)
            wt_std_mean = np.nanmean(wt_std_data)
            
            wt_mean.append(wt_std_mean)
            bt_mean.append(bt_std_mean)

            # print('Std of diff of unthresholded tstats in Subject {} \n{}:\nBT {}\nWT {}'.format('%.2d' % i, pair_, bt_std_mean, wt_std_mean))
            # print("stop")
        avg_bt_wt = (mean(wt_mean) + mean(bt_mean))/2
        if avg_bt_wt < min_sbj:
            min_sbj = avg_bt_wt
            i_min = i
        if avg_bt_wt > max_sbj:
            max_sbj = avg_bt_wt
            i_max = i
        # print('{}: bt_mean {} and wt_mean {}'.format(pair_, mean(bt_mean), mean(wt_mean)))
    print('i_min {} std {} \ni_max {} std {}'.format(i_min, min_sbj, i_max, max_sbj))
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
    ### Plot correlation of variances between BT and FL (Fig 4)
    # plot_corr_variances()

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
    # print_gl_stats(tool_results, mca_results)
    # print_sl_stats(tool_results, mca_results)


if __name__ == '__main__':
    main()