"""Andriy feature-selected SVM branch using a pinned R runtime."""
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
suppressPackageStartupMessages(library(LiblineaR))
stopifnot(as.character(getRversion())=='4.6.1', as.character(packageVersion('xgboost'))=='3.2.1.1', as.character(packageVersion('R.matlab'))=='3.8.0', as.character(packageVersion('LiblineaR'))=='2.10.24')
args <- commandArgs(trailingOnly=TRUE)
data <- readMat(args[1])
all_scores <- NULL
all_windows <- NULL
set.seed(1234)
for (i in 1:3) {
  train <- data[[paste0('train',i)]]
  labels <- as.numeric(data[[paste0('labels',i)]])
  prediction <- data[[paste0('prediction',i)]]
  valid <- as.logical(data[[paste0('valid',i)]])
  dtrain <- xgb.DMatrix(train,label=labels,nthread=1)
  params <- list(objective='binary:logistic',booster='gbtree',eval_metric='auc',eta=.3,max_depth=3,subsample=.8,colsample_bytree=1,num_parallel_tree=2,nthread=1)
  selector <- xgb.train(params=params,data=dtrain,nrounds=1000,verbose=0)
  importance <- xgb.importance(model=selector,feature_names=as.character(seq_len(ncol(train))))
  stopifnot(nrow(importance)>=300)
  indices <- as.integer(importance$Feature[1:300])
  prediction[is.na(prediction)] <- 0
  scaled <- scale(train[,indices],center=TRUE,scale=TRUE)
  stopifnot(all(is.finite(scaled)))
  model <- LiblineaR(data=scaled,target=as.factor(labels),type=5,cost=100,bias=TRUE,verbose=FALSE)
  stopifnot(as.character(model$ClassNames[1])=='1')
  test <- scale(prediction[,indices],attr(scaled,'scaled:center'),attr(scaled,'scaled:scale'))
  decisions <- predict(model,test,decisionValues=TRUE)$decisionValues[,1]
  windows <- 1/(1+exp(-decisions))
  masked <- numeric(length(windows))
  masked[valid] <- windows[valid]
  segments <- apply(matrix(masked,nrow=19),2,max)
  wholly_invalid <- colSums(matrix(valid,nrow=19))==0
  segments[wholly_invalid] <- 0
  segments[is.na(segments)] <- 0
  all_scores <- c(all_scores,segments)
  all_windows <- c(all_windows,windows)
}
writeMat(args[2],scores=all_scores,windows=all_windows)
'''


def witness_andriy_r_svm_segment_probabilities(training_features,training_labels,prediction_features,valid_prediction_rows):
    count=sum(x.shape[0]//19 for x in prediction_features)
    return (AbstractArray(shape=(count,),dtype='float64'),AbstractArray(shape=(sum(x.shape[0] for x in prediction_features),),dtype='float64'))


@register_atom(witness_andriy_r_svm_segment_probabilities)
def andriy_r_svm_segment_probabilities(training_features: list,training_labels: list,prediction_features: list,valid_prediction_rows: list) -> tuple[NDArray[np.float64],NDArray[np.float64]]:
    """Fit source SVMs for three populations and return segment/window scores.

    Input lists contain normalized windows/1965 feature matrices, binary labels
    and boolean prediction validity. Require positive-first training rows as in
    source P/P1/I stacking, finite training features, and prediction rows in
    complete groups of 19. Prediction NaNs become zero. A 1000-round source
    XGBoost selector must use at least 300 features. Explicit one-based numeric
    feature names adapt the current R importance API. Scale selected columns
    with training means/sample SD; fit LiblineaR type5, cost100, bias TRUE.
    Apply the source sigmoid to first-column decisions (checked as class1).
    Mask invalid windows to zero before 19-window maxima, without final ranks.

    Single R seed1234 precedes the three-population loop. R4.6.1, XGBoost3.2.1.1,
    LiblineaR2.10.24 and R.matlab3.8.0 are required. SCIONA_RSCRIPT/R_LIBS_USER
    configure the external runtime. Current defaults and serial execution;
    historical runtime equivalence is unproven. No persisted input/model files.
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
        if labels[0]!=1 or np.any(np.diff(labels)>0):
            raise ValueError('source positive-first training class order required')
        total_segments+=len(prediction)//19
        payload.update({f'train{i}':train.astype(float),f'labels{i}':labels.astype(float),
                        f'prediction{i}':prediction.astype(float),f'valid{i}':valid.astype(np.uint8)})
    executable=os.environ.get('SCIONA_RSCRIPT') or shutil.which('Rscript')
    if not executable:
        raise RuntimeError('Rscript is required for the Andriy R SVM provider')
    with TemporaryDirectory(prefix='sciona-andriy-r-svm-') as temporary:
        directory=Path(temporary)
        savemat(directory/'input.mat',payload)
        (directory/'run.R').write_text(_R_SCRIPT)
        try:
            run=subprocess.run([executable,'--vanilla',str(directory/'run.R'),str(directory/'input.mat'),str(directory/'output.mat')],
                env=dict(os.environ,OMP_NUM_THREADS='1'),capture_output=True,timeout=1200)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError('Andriy R SVM execution timed out') from exc
        if run.returncode:
            raise RuntimeError('Andriy R SVM execution failed; check the pinned R runtime and packages')
        output=loadmat(directory/'output.mat')
    scores=output['scores'].reshape(-1)
    windows=output['windows'].reshape(-1)
    if scores.shape!=(total_segments,) or windows.shape!=(total_segments*19,) or not np.all(np.isfinite(scores)) or not np.all(np.isfinite(windows)):
        raise RuntimeError('invalid Andriy R SVM output')
    return scores,windows
