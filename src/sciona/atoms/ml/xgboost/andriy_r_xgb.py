"""Andriy five-model XGBoost branch using a pinned R runtime."""
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import numpy as np
from numpy.typing import NDArray
from scipy.io import savemat,loadmat
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


_R_SCRIPT = r'''
suppressPackageStartupMessages(library(xgboost))
suppressPackageStartupMessages(library(R.matlab))
stopifnot(as.character(getRversion())=='4.6.1', as.character(packageVersion('xgboost'))=='3.2.1.1', as.character(packageVersion('R.matlab'))=='3.8.0')
args <- commandArgs(trailingOnly=TRUE)
data <- readMat(args[1])
all_segments <- NULL
seeds <- c(1234,23,13123,123131,1123123134)
for (i in 1:3) {
  train <- data[[paste0('train',i)]]
  labels <- as.numeric(data[[paste0('labels',i)]])
  prediction <- data[[paste0('prediction',i)]]
  valid <- as.logical(data[[paste0('valid',i)]])
  prediction[is.na(prediction)] <- 0
  dtrain <- xgb.DMatrix(train,label=labels,nthread=1)
  dtest <- xgb.DMatrix(prediction,nthread=1)
  params <- list(objective='binary:logistic',booster='gbtree',eval_metric='auc',eta=.3,max_depth=3,subsample=.8,colsample_bytree=1,num_parallel_tree=2,nthread=1)
  windows <- matrix(0,nrow(prediction),5)
  for (j in 1:5) {
    set.seed(seeds[j])
    model <- xgb.train(params=params,data=dtrain,nrounds=1000,verbose=0)
    windows[,j] <- predict(model,dtest)
  }
  segments <- matrix(0,nrow(prediction)/19,5)
  for (j in 1:5) segments[,j] <- apply(matrix(windows[,j],nrow=19),2,max)
  wholly_invalid <- colSums(matrix(valid,nrow=19))==0
  segments[wholly_invalid,] <- 0
  all_segments <- rbind(all_segments,segments)
}
ranks <- apply(all_segments,2,rank,ties.method='average')/nrow(all_segments)
scores <- apply(ranks,1,function(x) mean(x))
writeMat(args[2],scores=scores,segments=all_segments)
'''


def witness_andriy_r_xgb_segment_probabilities(training_features,training_labels,prediction_features,valid_prediction_rows):
    count=sum(x.shape[0]//19 for x in prediction_features)
    return (AbstractArray(shape=(count,),dtype='float64'),AbstractArray(shape=(count,5),dtype='float64'))


@register_atom(witness_andriy_r_xgb_segment_probabilities)
def andriy_r_xgb_segment_probabilities(training_features: list,training_labels: list,prediction_features: list,valid_prediction_rows: list) -> tuple[NDArray[np.float64],NDArray[np.float64]]:
    """Train the source five seeded models for each of three ordered populations.

    Lists have exactly three entries. Training/prediction matrices contain 1965
    already normalized features; train rows are finite, both binary labels are
    required, and prediction NaNs are replaced by zero. Prediction row counts
    must be positive multiples of 19 with boolean validity masks. Each model's
    probabilities are maximized across all 19 windows, including invalid windows;
    only wholly invalid segments become zero before ranking. Average-tie ranks
    are computed across all populations separately for each model, divided by
    total segments, then averaged. Invalid segments can thus have nonzero ranks.

    Return concatenated rank scores and the pre-rank segments/5 probabilities.
    R 4.6.1, xgboost 3.2.1.1 and R.matlab 3.8.0 are required. SCIONA_RSCRIPT may
    select the executable; R_LIBS_USER configures the external library. Current
    engine defaults apply with serial execution. No historical engine identity
    or predictive quality claim. Runtime data stays in a temporary directory;
    no model files, dataset identifiers or captured R diagnostics are published.
    """
    groups=[training_features,training_labels,prediction_features,valid_prediction_rows]
    if any(not isinstance(group,list) or len(group)!=3 for group in groups):
        raise ValueError('exactly three ordered populations required in each input list')
    payload={}
    total_segments=0
    for i,(train,labels,prediction,valid) in enumerate(zip(*groups),1):
        train,labels,prediction,valid=map(np.asarray,[train,labels,prediction,valid])
        for value in [train,prediction]:
            if value.dtype.kind not in 'iuf' or value.ndim!=2 or value.shape[0]==0 or value.shape[1]!=1965 or np.any(np.isinf(value)):
                raise ValueError('nonempty real windows/1965 feature matrices without infinities required')
        if not np.all(np.isfinite(train)):
            raise ValueError('training features must be finite')
        if labels.dtype.kind not in 'iuf' or labels.shape!=(len(train),) or not np.array_equal(np.unique(labels),[0,1]):
            raise ValueError('one binary training label per row and both classes required')
        if len(prediction)%19 or valid.dtype.kind!='b' or valid.shape!=(len(prediction),):
            raise ValueError('prediction rows must form complete 19-window segments with boolean validity')
        total_segments+=len(prediction)//19
        payload.update({f'train{i}':train.astype(float),f'labels{i}':labels.astype(float),
                        f'prediction{i}':prediction.astype(float),f'valid{i}':valid.astype(np.uint8)})
    executable=os.environ.get('SCIONA_RSCRIPT') or shutil.which('Rscript')
    if not executable:
        raise RuntimeError('Rscript is required for the Andriy R XGBoost provider')
    with TemporaryDirectory(prefix='sciona-andriy-r-xgb-') as temporary:
        directory=Path(temporary)
        savemat(directory/'input.mat',payload)
        (directory/'run.R').write_text(_R_SCRIPT)
        try:
            run=subprocess.run([executable,'--vanilla',str(directory/'run.R'),str(directory/'input.mat'),str(directory/'output.mat')],
                env=dict(os.environ,OMP_NUM_THREADS='1'),capture_output=True,timeout=1200)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError('Andriy R XGBoost execution timed out') from exc
        if run.returncode:
            raise RuntimeError('Andriy R XGBoost execution failed; check the pinned R runtime and packages')
        output=loadmat(directory/'output.mat')
    scores=output['scores'].reshape(-1)
    segments=output['segments']
    if scores.shape!=(total_segments,) or segments.shape!=(total_segments,5) or not np.all(np.isfinite(scores)) or not np.all(np.isfinite(segments)):
        raise RuntimeError('invalid Andriy R XGBoost output')
    return scores,segments
