import math
import typing as T
from typing import Callable, Any

import numpy as np
import numpy.typing as nT
import pandas as pd
import torch
import torch as th
from scipy.stats import wilcoxon, mannwhitneyu # type: ignore
from sklearn import metrics as M

from helper_functions import prc as prc
Tensor = torch.Tensor
NDarray = nT.NDArray
ndarray = np.ndarray



def AuPRC_torch(targs: Tensor, preds: Tensor):
    pr, rc, _ = prc.prc_torch(targs.flatten(), preds.flatten())
    idx = torch.argsort(rc)
    pr, rc = pr[idx], rc[idx]
    return 100 * torch.trapz(pr, rc)


def fmax_torch(targs: Tensor, preds: Tensor):
    n = 100

    fmax_score = torch.as_tensor(0.)
    for t in range(n+1):
        threshold = t / n

        pred_bi = torch.where(preds > threshold,1,0)
        # score = M.f1_score(targs.flatten(), pred_bi.flatten())

        # MCM = M.multilabel_confusion_matrix(targs, pred_bi)
        tp = (pred_bi * targs).sum(1)
        tp_and_fp = pred_bi.sum(1)
        tp_and_fn = targs.sum(1)

        idx = torch.where(tp_and_fp != 0)
        tp_and_fp = tp_and_fp[idx]

        # control zero division
        precision = (tp[idx[0]] / tp_and_fp
        if tp_and_fp.size(0) != 0
        else torch.as_tensor(0.0)).mean()

        idx = torch.where(tp_and_fn != 0)
        tp_and_fn = tp_and_fn[idx]
        recall = (tp[idx] / tp_and_fn
        if tp_and_fn.size(0) != 0
        else torch.as_tensor(0.0)).mean()

        if (denom := precision + recall) == 0.0:
            denom = torch.as_tensor(1.)
        if (score := 2 * precision * recall / denom) > fmax_score:
            fmax_score = score
    return fmax_score * 100

class PReprot(T.TypedDict):
    fmax: float
    threshold: float
    smin: float
    auprc: float

def evalperf_torch(targs: th.Tensor, 
                   preds: th.Tensor,
                   threshold: bool=False,
                   smin: bool=False,
                   auprc: bool=False,
                   no_empty_labels: bool=False,
                   no_zero_classes: bool=False,
                   icary: th.Tensor | None=None,
                   ):
    n = 100
    if no_empty_labels:
        idx = torch.where(targs.sum(1))[0]
        targs = targs[idx]
        preds = preds[idx]
    
    if no_zero_classes:
        idx = targs.sum(0) > 0
        targs = targs[:, idx]
        preds = preds[:, idx]
    
    assert not smin or isinstance(icary, torch.Tensor), \
    "when smin is true, icary must be not none"

    report: PReprot
    report = {"fmax": 0., "threshold": 0., "smin": -1.,
              "auprc": 0.}
    mi, ru = 0., 0.
    prs = th.zeros(n+1)
    rcs = th.zeros(n+1)
    for i, t in enumerate(range(1, n+1)):
        thres = t / n

        pred_mask = preds > thres
        targ_mask = targs > 0
        pred_bi = torch.where(pred_mask, 1, 0)
        tpM = pred_bi * targs
        fnM = targs * torch.where(~pred_mask, 1, 0)
        fpM = torch.where(~targ_mask, 1, 0) * pred_bi

        tp_sum = tpM.sum().item()
        pred_sum = pred_bi.sum().item()
        true_sum = targs.sum().item()

        # control zero division
        precision = tp_sum / pred_sum if pred_sum != 0.0 else 0.0
        recall = tp_sum / true_sum if true_sum != 0.0 else 0.0

        # fmax
        denom = precision + recall
        if denom == 0.0: denom = 1
        score = 100 * 2 * precision * recall / denom
        if score > report["fmax"]:
            report["fmax"] = score
            report["threshold"] = thres

        # mi, ru, smin
        if icary is not None:
            mi = (fpM * icary).sum(1).mean().item()
            ru = (fnM * icary).sum(1).mean().item()
        smin_score = math.sqrt(ru*ru+mi*mi)
        if report["smin"] < 0 or smin_score < report["smin"]:
            report["smin"] = smin_score

        # precision, recall
        prs[i] = precision
        rcs[i] = recall

    sorted_index = torch.argsort(rcs)
    report["auprc"] = torch.trapz(prs[sorted_index],
                                  rcs[sorted_index]) * 100
    keys = ["fmax"]
    if threshold: keys.append("threshold")
    if smin: keys.append("smin")
    if auprc: keys.append("auprc")
    return {k: report[k] for k in keys}


def AuPRC_score(targs: np.ndarray, preds: np.ndarray):
    pr_ary, rc_ary, _ = M.precision_recall_curve(targs.ravel(), preds.ravel())
    AuPRC_value = M.auc(rc_ary, pr_ary)
    return 100 * AuPRC_value


def fmax_score(targs: np.ndarray, preds: np.ndarray,
               no_empty_labels: bool = True,
               need_threshold: bool = False,
               auprc: bool = False,
               curve: bool = False):
    n = 100
    if no_empty_labels:
        idx = np.where(targs.sum(1) != 0)
        targs = targs[idx]
        preds = preds[idx]
    fmax_value = 0.
    max_thres = 0.

    pr, rc = [], []
    for t in range(n+1):
        threshold = t / n
        preds_label = np.where(preds > threshold, 1, 0)
        tp = (preds_label * targs).sum(1)
        tp_fp = preds_label.sum(1)
        tp_fn = targs.sum(1)

        idx= np.where(tp_fp != 0)
        tp_fp = tp_fp[idx]
        precision = (tp[idx] / tp_fp
        if tp_fp.shape[0] != 0
        else np.asarray([0.])).mean()

        # when no_empty_label is False
        idx = np.where(tp_fn != 0)
        tp_fn = tp_fn[idx]
        recall = (tp[idx] / tp_fn
        if tp_fn.shape[0] != 0
        else np.asarray([0.])).mean()

        if ((denom := precision + recall) == 0.0):
            denom = 1.0
        if (score := 2 * precision * recall / denom) > fmax_value:
          fmax_value = score
          max_thres = threshold
        
        if t == 0:
          pr.append(0.)
        elif t < n:
          pr.append(precision)
        else: # t == n
          pass
        
        if t == n:
          rc.append(0.)
          pr.append(1.)
        elif pr[-1] <= 0:
          rc.append(1.)
        else:
          rc.append(recall)

    # fmax_value = reduce(_calculate_fmax, np.arange(n+1))
    # fmax_value = max(map(_calculate_fmax, np.arange(n+1)))
    pr = np.array(pr)
    rc = np.array(rc)
    index = np.argsort(rc)

    if need_threshold and curve:
      return (fmax_value * 100, max_thres), (pr, rc)
    elif need_threshold and auprc:
      return (fmax_value * 100, max_thres), np.trapz(pr[index], rc[index])
    elif need_threshold:
      return fmax_value * 100, max_thres
    elif curve:
      return (fmax_value * 100, ), (pr, rc)
    elif auprc:
      return (fmax_value * 100,), 100 * np.trapz(pr[index], rc[index])
    else:
      return fmax_value * 100


def fmax_sklearn(targs: ndarray, preds: ndarray):
    """
    """
    n = 100
    # return max(f1_at(x/n) for x in range(n+1)) * 100
    targs = targs.astype(int)
    fmax_score = 0.
    for t in range(n+1):
        threshold = t / n

        pred_bi = np.where(preds > threshold,1,0)
        # score = M.f1_score(targs.flatten(), pred_bi.flatten())

        # MCM = M.multilabel_confusion_matrix(targs, pred_bi)
        tp_sum = np.logical_and(targs, pred_bi).sum()
        pred_sum = pred_bi.sum()
        true_sum = targs.sum()

        # control zero division
        precision = tp_sum / pred_sum if pred_sum != 0.0 else 0.0
        recall = tp_sum / true_sum if true_sum != 0.0 else 0.0

        denom = precision + recall
        if denom == 0.0: denom = 1
        if (score := 2 * precision * recall / denom) > fmax_score:
            fmax_score = score
    return fmax_score * 100

def fmax_by_curve(targs: np.ndarray, preds: np.ndarray,
                  need_threshold: bool = False):
  pr, rc, th = M.precision_recall_curve(targs.ravel(), preds.ravel())
  fxs = (2 * pr * rc) / (pr + rc)
  i = fxs.argmax()
  
  if need_threshold:
    return 100 * fxs[i], th[i]
  else:
    return 100 * fxs[i]

def APscore(targs: np.ndarray, preds: np.ndarray):
  pr, rc, _ = M.precision_recall_curve(targs.ravel(), preds.ravel())
  AP = np.sum((rc[:-1] - rc[1:]) * pr[:-1])
  return AP * 100

def AUCscore(targs: np.ndarray, preds: np.ndarray):
   fpr, tpr, _ = M.roc_curve(targs.flatten(), preds.flatten())
   score = M.auc(fpr, tpr)
   return score * 100


def simple_prf_divide(numerator: NDarray, denominator: NDarray):
    mask = denominator == 0.0
    denominator = denominator.copy()
    denominator[mask] = 1  # avoid infs/nans
    result = np.true_divide(numerator, denominator)

    if not np.any(mask):
        return result
    result[mask] = 0.0
    return result


def fmax_tp(tp: np.ndarray, no_empty_labels: bool = True):
    """
    tp: 2 x 1 x n_classes
    """
    t, p = tp # both are 1 x n_classes
    return fmax_score(t,p, no_empty_labels)


def AuPRC_tp(tp: ndarray):
    t, p = tp
    return AuPRC_score(t,p)


def fmax_pvalue(pvaluefunc: Callable[[ndarray,ndarray, str], Any]):
    def _wrapper(targs_preds0, targs_preds1,
                 alternative: str = "two_sided",
                 no_empty_labels:bool=False):
        vfmax_score = np.vectorize(fmax_tp,signature="(i,j,k),()->()")
        fm_ary0 = vfmax_score(targs_preds0, no_empty_labels)
        fm_ary1 = vfmax_score(targs_preds1, no_empty_labels)

        _, pvalue = pvaluefunc(fm_ary0, fm_ary1, alternative)
        return pvalue
    return _wrapper


def AuPRC_pvalue(pvaluefunc: Callable[[ndarray,ndarray, str], Any]):
    def _wrapper(targs_preds0, targs_preds1,
                 alternative: str = "two_sided"):
        vAuPRC_score = np.vectorize(AuPRC_tp, signature="(i,j,k)->()")

        au_ary0 = vAuPRC_score(targs_preds0)
        au_ary1 = vAuPRC_score(targs_preds1)

        _, pvalue = pvaluefunc(au_ary0, au_ary1, alternative)
        return pvalue
    return _wrapper


@fmax_pvalue
def fmax_wilcoxon(m0: ndarray,m1: ndarray, alternative: str = "two_sided"):
    return wilcoxon(m0,m1, alternative=alternative)


@AuPRC_pvalue
def AuPRC_wilcoxon(m0: ndarray, m1: ndarray, alternative: str = "two_sided"):
    return wilcoxon(m0, m1, alternative=alternative)


@fmax_pvalue
def fmax_mannwhitneyu(m0: ndarray, m1: ndarray, alternative: str = "two_sided"):
    return mannwhitneyu(m0, m1, alternative=alternative)


@AuPRC_pvalue
def AuPRC_mannwhitneyu(m0: ndarray, m1:ndarray, alternative: str = "two_sided"):
    return mannwhitneyu(m0, m1, alternative=alternative)

def eval_performance(targs: np.ndarray, preds: np.ndarray,
                     threshold: bool = False,
                     smin: bool = False,
                     auprc: bool = False,
                     no_empty_labels: bool = False,
                     no_zero_classes: bool = False,
                     icary: np.ndarray | None = None,
                     ):
    n = 100
    if no_empty_labels:
        idx = np.where(targs.sum(1))[0]
        targs = targs[idx]
        preds = preds[idx]
    
    if no_zero_classes:
        idx = targs.sum(0) > 0
        targs = targs[:, idx]
        preds = preds[:, idx]
        icary = icary[idx]
    
    assert not smin or isinstance(icary, np.ndarray), \
    "when smin is true, icary must be not none"

    report: PReprot
    report = {"fmax": 0., "threshold": 0., "smin": -1.,
              "auprc": 0.}
    mi, ru = 0., 0.
    prs = np.zeros(n+1)
    rcs = np.zeros(n+1)
    for i, t in enumerate(range(1, n+1)):
        thres = t / n
        # preds_label = np.where(preds > thres, 1, 0)
        # true_positive = np.where(np.logical_and(preds_label, targs), 1, 0)
        pred_mask = preds > thres
        targ_mask = targs > 0
        pred_bi = np.where(pred_mask, 1, 0)
        tpM = pred_bi * targs
        fnM = targs * np.where(~pred_mask, 1, 0) # 1 * (0 -> 1), true but false
        fpM = np.where(~targ_mask, 1, 0) * pred_bi # (0 -> 1) * 1, false but true

        tp_sum = tpM.sum()
        pred_sum = pred_bi.sum()
        true_sum = targs.sum()

        # control zero division
        precision = tp_sum / pred_sum if pred_sum != 0.0 else 0.0
        recall = tp_sum / true_sum if true_sum != 0.0 else 0.0

        # fmax
        denom = precision + recall
        if denom == 0.0: denom = 1
        if (score := 100 * 2 * precision * recall / denom) > report["fmax"]: 
            report["fmax"] = score
            report["threshold"] = thres

        # mi, ru, smin
        if icary is not None:
            mi = (fpM * icary).sum(1).mean()
            ru = (fnM * icary).sum(1).mean()
        smin_score = np.sqrt(ru*ru+mi*mi)
        if report["smin"] < 0 or smin_score < report["smin"]:
            report["smin"] = smin_score

        # precision, recall
        prs[i] = precision
        rcs[i] = recall

    sorted_index = np.argsort(rcs)
    report["auprc"] = np.trapz(prs[sorted_index],
                               rcs[sorted_index]) * 100
    keys = ["fmax"]
    if threshold: keys.append("threshold")
    if smin: keys.append("smin")
    if auprc: keys.append("auprc")
    return {k: report[k] for k in keys}

def evaluate_by(nspace_ti: T.Dict[str, T.Dict[str, int]],
                term_count_ic: pd.DataFrame,
                by: str,
                key: str,
                ont_pred: np.ndarray,
                low: T.Union[float, int, None]=None, 
                high: T.Union[float, int, None]=None):
  targs, preds = ont_pred
  if low is None and high is None:
    return fmax_score(targs, preds, need_threshold=True), AuPRC_score(targs, preds)
  
  assert low is not None or high is not None
  if low is not None and high is not None:
    assert high > low

  assert key in term_count_ic.columns

  index = index_of_term(nspace_ti, term_count_ic, by, key, low, high)
  
  return fmax_score(targs[:, index], preds[:, index], need_threshold=True), \
      AuPRC_score(targs[:, index], preds[:, index])


def index_of_term(nspace_ti: T.Dict[str, T.Dict[str, int]],
                  term_count_ic: pd.DataFrame,
                  by: str,
                  key: str,
                  low: float | int | None = None,
                  high: float | int | None = None):
    if high is None:
        index = term_count_ic[term_count_ic[key] >= low]["annotation"] \
            .transform(lambda x: nspace_ti[by].get(x, pd.NA)) \
            .dropna().to_numpy(dtype=int)
    elif low is None:
        index = term_count_ic[term_count_ic[key] < high]["annotation"] \
            .transform(lambda x: nspace_ti[by].get(x, pd.NA)) \
            .dropna().to_numpy(dtype=int)
    else:
        index = term_count_ic[(term_count_ic[key] >= low) &
                              (term_count_ic[key] < high)]["annotation"] \
            .transform(lambda x: nspace_ti[by].get(x, pd.NA)) \
            .dropna().to_numpy(dtype=int)
    return index
