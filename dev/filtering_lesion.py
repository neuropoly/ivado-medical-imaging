#!/usr/bin/env python
##############################################################
#
# TODO
#
##############################################################

import os
import json
import joblib
import argparse
import pandas as pd
import numpy as np
import nibabel as nib
from copy import deepcopy
# from sklearn.metrics import auc
import matplotlib.pyplot as plt
from scipy.ndimage import label, generate_binary_structure
from loguru import logger

from ivadomed import config_manager as imed_config_manager
from ivadomed import main as imed
from ivadomed import utils as imed_utils
from ivadomed import metrics as imed_metrics
from ivadomed import postprocessing as imed_postpro

BIN_STRUCT = generate_binary_structure(3, 2)
MIN_OBJ_SIZE = 3

# experiments
exp_dct = {
    'exp1': {'level': 'vox',
             'uncertainty_measure': '_unc-vox',
             'uncertainty_thr': [1e-3, 1e-2, 0.1, 0.2, 0.4, 0.6],
             'prediction_thr': [t / 10. for t in range(1, 10, 1)]}
    #                'exp2': {'level': 'obj',
    #                            'uncertainty_measure': '_unc-cv',
    #                            'uncertainty_thr': [0.1, 0.2, 0.3, 0.4, 0.5],
    #                            'prediction_thr': [1e-6]+[t/10. for t in range(1,10,1)]},
    #                'exp3': {'level': 'obj',
    #                            'uncertainty_measure': '_unc-avgUnc',
    #                            'uncertainty_thr': [0.1, 0.2, 0.3, 0.4, 0.5],
    #                            'prediction_thr': [1e-6]+[t/10. for t in range(1,10,1)]}
}


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", help="Config file path.")
    parser.add_argument("-thrUnc", help="Threshold to apply on uncertainty map.", default=None)
    parser.add_argument("-thrPred", help="Threshold to apply on prediction.", default=None)
    parser.add_argument("-suffixUnc", help="Suffix of the uncertainty map to use (e.g. _unc-vox).", default=None)
    parser.add_argument("-ofolder", help="Output folder.")

    return parser


def auc_homemade(fpr, tpr, trapezoid=False):
    # source: https://stackoverflow.com/a/39687168
    inds = [i for (i, (s, e)) in enumerate(zip(fpr[: -1], fpr[1:])) if s != e] + [len(fpr) - 1]
    fpr, tpr = fpr[inds], tpr[inds]
    area = 0
    ft = list(zip(fpr, tpr))
    for p0, p1 in zip(ft[: -1], ft[1:]):
        area += (p1[0] - p0[0]) * ((p1[1] + p0[1]) / 2 if trapezoid else p0[1])
    return -area


def print_unc_stats(unc_name, pred_folder, im_lst):
    mins, p25s, p50s, p75s, maxs = [], [], [], [], []
    for fname_pref in im_lst:
        fname_unc = os.path.join(pred_folder, fname_pref + unc_name + '.nii.gz')
        im = nib.load(fname_unc)
        data_unc = im.get_data()
        del im
        vals = list(data_unc[data_unc > 0])
        if len(vals):
            mins.append(np.min(vals))
            maxs.append(np.max(vals))
            p25s.append(np.percentile(vals, 25))
            p50s.append(np.percentile(vals, 50))
            p75s.append(np.percentile(vals, 75))

    for n, l in zip(['min', 'max', 'p25', 'p50', 'p75'], [mins, maxs, p25s, p50s, p75s]):
        logger.debug(f"\t{n}: {np.mean(l)}")


def count_retained(data_before, data_after, level):
    if level == 'vox':
        cmpt_before, cmpt_after = np.count_nonzero(data_before), np.count_nonzero(data_after)
    else:  # level == 'obj'
        logger.debug(f"{np.sum(data_before)} {np.sum(data_after)}")
        _, cmpt_before = label(data_before, structure=BIN_STRUCT)
        _, cmpt_after = label(data_after, structure=BIN_STRUCT)
        logger.debug(f"{cmpt_before} {cmpt_after}")
    percent_rm = (cmpt_before - cmpt_after) * 100. / cmpt_before
    return 100. - percent_rm


def run_experiment(level, unc_name, thr_unc_lst, thr_pred_lst, gt_folder, pred_folder, im_lst, target_suf, param_eval):
    # init results
    tmp_lst = [[] for _ in range(len(thr_pred_lst))]
    res_init_lst = [deepcopy(tmp_lst) for _ in range(len(thr_unc_lst))]
    res_dct = {'tpr': deepcopy(res_init_lst),
               'fdr': deepcopy(res_init_lst),
               'retained_elt': [[] for _ in range(len(thr_unc_lst))]
               }

    # loop across images
    for fname_pref in im_lst:
        # uncertainty map
        fname_unc = os.path.join(pred_folder, fname_pref + unc_name + '.nii.gz')
        im = nib.load(fname_unc)
        data_unc = im.get_data()
        del im

        # list MC samples
        data_pred_lst = np.array([nib.load(os.path.join(pred_folder, f)).get_data()
                                  for f in os.listdir(pred_folder) if fname_pref + '_pred_' in f])

        # ground-truth fname
        fname_gt = os.path.join(gt_folder, fname_pref.split('_')[0], 'anat', fname_pref + target_suf + '.nii.gz')
        if os.path.isfile(fname_gt):
            nib_gt = nib.load(fname_gt)
            data_gt = nib_gt.get_data()
            logger.debug(np.sum(data_gt))
            # soft prediction
            data_soft = np.mean(data_pred_lst, axis=0)

            if np.any(data_soft):
                for i_unc, thr_unc in enumerate(thr_unc_lst):
                    # discard uncertain lesions from data_soft
                    data_soft_thrUnc = deepcopy(data_soft)
                    data_soft_thrUnc[data_unc > thr_unc] = 0
                    cmpt = count_retained((data_soft > 0).astype(np.int), (data_soft_thrUnc > 0).astype(np.int), level)
                    res_dct['retained_elt'][i_unc].append(cmpt)
                    logger.debug(f"{thr_unc} {cmpt}")
                    for i_pred, thr_pred in enumerate(thr_pred_lst):
                        data_hard = imed_postpro.threshold_predictions(deepcopy(data_soft_thrUnc), thr=thr_pred)\
                                                .astype(np.uint8)

                        eval = imed_utils.Evaluation3DMetrics(data_pred=data_hard,
                                                              data_gt=data_gt,
                                                              dim_lst=nib_gt.header['pixdim'][1:4],
                                                              params=param_eval)

                        if level == 'vox':
                            tpr = imed_metrics.recall_score(eval.data_pred, eval.data_gt, err_value=np.nan)
                            fdr = 100. - imed_metrics.precision_score(eval.data_pred, eval.data_gt, err_value=np.nan)
                        else:
                            tpr, _ = eval.get_ltpr()
                            fdr = eval.get_lfdr()
                        logger.debug(f"{thr_pred} {np.count_nonzero(deepcopy(data_soft_thrUnc))} "
                                     f"{np.count_nonzero(data_hard)} {tpr} {fdr}")
                        res_dct['tpr'][i_unc][i_pred].append(tpr / 100.)
                        res_dct['fdr'][i_unc][i_pred].append(fdr / 100.)

    return res_dct


def print_retained_elt(thr_unc_lst, retained_elt_lst):
    logger.info('Mean percentage of retained elt:')
    for i, t in enumerate(thr_unc_lst):
        logger.info(f"\tUnc threshold: {t} --> {np.mean(retained_elt_lst[i])}")


def plot_roc(thr_unc_lst, thr_pred_lst, res_dct, metric, fname_out):
    plt.figure(figsize=(10, 10))
    for i_unc, thr_unc in enumerate(thr_unc_lst):
        logger.info(f"Unc Thr: {thr_unc}")

        tpr_vals = np.array([np.nanmean(res_dct['tpr'][i_unc][i_pred]) for i_pred in range(len(thr_pred_lst))])
        fdr_vals = np.array([np.nanmean(res_dct['fdr'][i_unc][i_pred]) for i_pred in range(len(thr_pred_lst))])

        auc_ = auc_homemade(fdr_vals, tpr_vals, True)

        optimal_idx = np.argmax(tpr_vals - fdr_vals)
        optimal_threshold = thr_pred_lst[optimal_idx]
        logger.info(f"AUC: {auc_}, Optimal Pred Thr: {optimal_threshold}")

        plt.scatter(fdr_vals, tpr_vals, label='Unc thr={0:0.2f} (area = {1:0.2f})'.format(thr_unc, auc_), s=22)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Detection Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC - ' + metric)
    plt.legend(loc="lower right")
    plt.savefig(fname_out, bbox_inches='tight', pad_inches=0)
    plt.close()


def run_inference(pred_folder, im_lst, thr_pred, gt_folder, target_suf, param_eval, unc_name=None, thr_unc=None):
    # init df row list
    df_lst = []

    # loop across images
    for fname_pref in im_lst:
        if not any(elem is None for elem in [unc_name, thr_unc]):
            logger.debug(thr_unc)
            # uncertainty map
            fname_unc = os.path.join(pred_folder, fname_pref + unc_name + '.nii.gz')
            im = nib.load(fname_unc)
            data_unc = im.get_data()
            del im

            # list MC samples
            data_pred_lst = np.array([nib.load(os.path.join(pred_folder, f)).get_data()
                                      for f in os.listdir(pred_folder) if fname_pref + '_pred_' in f])
        else:
            data_pred_lst = np.array([nib.load(os.path.join(pred_folder, f)).get_data()
                                      for f in os.listdir(pred_folder) if fname_pref + '_pred.' in f])

        # ground-truth fname
        fname_gt = os.path.join(gt_folder, fname_pref.split('_')[0], 'anat', fname_pref + target_suf + '.nii.gz')
        nib_gt = nib.load(fname_gt)
        data_gt = nib_gt.get_data()

        # soft prediction
        data_soft = np.mean(data_pred_lst, axis=0)

        if not any(elem is None for elem in [unc_name, thr_unc]):
            logger.debug("thr")
            # discard uncertain lesions from data_soft
            data_soft[data_unc > thr_unc] = 0

        data_hard = imed_postpro.threshold_predictions(data_soft, thr=thr_pred).astype(np.uint8)

        eval = imed_utils.Evaluation3DMetrics(data_pred=data_hard,
                                              data_gt=data_gt,
                                              dim_lst=nib_gt.header['pixdim'][1:4],
                                              params=param_eval)

        results_pred, _ = eval.run_eval()

        # save results of this fname_pred
        results_pred['image_id'] = fname_pref.split('_')[0]
        df_lst.append(results_pred)

    df_results = pd.DataFrame(df_lst)
    return df_results


def run_main(args):
    thrPred = None if args.thrPred is None else float(args.thrPred)
    thrUnc = None if args.thrUnc is None else float(args.thrUnc)
    sufUnc = args.suffixUnc

    context = imed_config_manager.ConfigurationManager(args.c).get_config()

    ofolder = args.ofolder
    if not os.path.isdir(ofolder):
        os.makedirs(ofolder)

    pred_folder = os.path.join(context['path_output'], 'pred_masks')
    if not os.path.isdir(pred_folder):
        imed.cmd_test(context)

    subj_acq_lst = list(set([f.split('_pred')[0] for f in os.listdir(pred_folder)
                             if f.endswith('.nii.gz') and '_pred' in f]))
    # subj_acq_lst = [subj_acq_lst[0]]
    gt_folder = os.path.join(context['path_data'], 'derivatives', 'labels')

    if thrPred is None:
        for exp in exp_dct.keys():
            config_dct = exp_dct[exp]
            logger.debug(config_dct['uncertainty_measure'])

            # print_unc_stats is used to determine 'uncertainty_thr'
            print_unc_stats(config_dct['uncertainty_measure'], pred_folder, subj_acq_lst)

            res_ofname = os.path.join(ofolder, config_dct['uncertainty_measure'] + '.joblib')
            if not os.path.isfile(res_ofname):
                res = run_experiment(level=config_dct['level'],
                                     unc_name=config_dct['uncertainty_measure'],
                                     thr_unc_lst=config_dct['uncertainty_thr'],
                                     thr_pred_lst=config_dct['prediction_thr'],
                                     gt_folder=gt_folder,
                                     pred_folder=pred_folder,
                                     im_lst=subj_acq_lst,
                                     target_suf=context["target_suffix"][0],
                                     param_eval=context["eval_params"])
                joblib.dump(res, res_ofname)
            else:
                res = joblib.load(res_ofname)

            print_retained_elt(thr_unc_lst=config_dct['uncertainty_thr'], retained_elt_lst=res['retained_elt'])

            plot_roc(thr_unc_lst=config_dct['uncertainty_thr'],
                     thr_pred_lst=config_dct['prediction_thr'],
                     res_dct=res,
                     metric=config_dct['uncertainty_measure'],
                     fname_out=os.path.join(ofolder, config_dct['uncertainty_measure'] + '.png'))
    else:
        df = run_inference(pred_folder=pred_folder,
                           im_lst=subj_acq_lst,
                           thr_pred=thrPred,
                           gt_folder=gt_folder,
                           target_suf=context["target_suffix"][0],
                           param_eval=context["eval_params"],
                           unc_name=sufUnc,
                           thr_unc=thrUnc)
        logger.debug(df.head())
        vals = [v for v in df.dice_class0 if str(v) != 'nan']
        logger.info(f"Median (IQR): {np.median(vals)} ({np.percentile(vals, 25)} - {np.percentile(vals, 75)}).")
        df.to_csv(os.path.join(ofolder, '_'.join([str(sufUnc), str(thrUnc), str(thrPred)]) + '.csv'))


if __name__ == '__main__':
    parser = get_parser()
    arguments = parser.parse_args()
    run_main(arguments)
