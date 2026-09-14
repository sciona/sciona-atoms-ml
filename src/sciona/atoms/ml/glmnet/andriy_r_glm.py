"""Andriy three-model GLM branch using a pinned R runtime."""
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
suppressPackageStartupMessages(library(glmnet))
suppressPackageStartupMessages(library(LiblineaR))
stopifnot(as.character(getRversion())=='4.6.1', as.character(packageVersion('xgboost'))=='3.2.1.1', as.character(packageVersion('R.matlab'))=='3.8.0', as.character(packageVersion('LiblineaR'))=='2.10.24', as.character(packageVersion('glmnet'))=='5.0')
args <- commandArgs(trailingOnly=TRUE)
data <- readMat(args[1])
all_segments <- NULL
set.seed(1234)
for (i in 1:3) {
  train <- data[[paste0('train',i)]]
  labels <- as.numeric(data[[paste0('labels',i)]])
  prediction <- data[[paste0('prediction',i)]]
  valid <- as.logical(data[[paste0('valid',i)]])
  dtrain <- xgb.DMatrix(train,label=labels,nthread=1)
  params <- list(objective='binary:logistic',booster='gbtree',eval_metric='auc',eta=.3,max_depth=3,subsample=.8,colsample_bytree=1,num_parallel_tree=2,nthread=1)
  selector <- xgb.train(params=params,data=dtrain,nrounds=1500,verbose=0)
  importance <- xgb.importance(model=selector,feature_names=as.character(seq_len(ncol(train))))
  stopifnot(nrow(importance)>=200)
  indices <- as.integer(importance$Feature[1:200])
  prediction[is.na(prediction)] <- 0
  ridge200 <- glmnet(x=as.matrix(train[,indices]),y=labels,alpha=0,family='binomial')
  pred200 <- predict(ridge200,as.matrix(prediction[,indices]),s=.0001,type='response')
  ridge100 <- glmnet(x=as.matrix(train[,indices[1:100]]),y=labels,alpha=0,family='binomial')
  pred100 <- predict(ridge100,as.matrix(prediction[,indices[1:100]]),s=.0001,type='response')
  scaled <- scale(train[,indices],center=TRUE,scale=TRUE)
  stopifnot(all(is.finite(scaled)))
  model <- LiblineaR(data=scaled,target=as.factor(labels),type=6,cost=10,bias=TRUE,verbose=FALSE)
  stopifnot(as.character(model$ClassNames[1])=='1')
  test <- scale(prediction[,indices],attr(scaled,'scaled:center'),attr(scaled,'scaled:scale'))
  linear <- predict(model,test,proba=TRUE)$probabilities[,1]
  windows <- cbind(pred200,pred100,linear)
  segments <- matrix(0,nrow(prediction)/19,3)
  for (j in 1:3) segments[,j] <- apply(matrix(windows[,j],nrow=19),2,max)
  wholly_invalid <- colSums(matrix(valid,nrow=19))==0
  segments[wholly_invalid,] <- 0
  all_segments <- rbind(all_segments,segments)
}
ranks <- apply(all_segments,2,rank,ties.method='average')/nrow(all_segments)
scores <- apply(ranks,1,function(x) mean(x))
writeMat(args[2],scores=scores,segments=all_segments)
'''


def witness_andriy_r_glm_segment_probabilities(training_features,training_labels,prediction_features,valid_prediction_rows):
    count=sum(x.shape[0]//19 for x in prediction_features)
    return (AbstractArray(shape=(count,),dtype='float64'),AbstractArray(shape=(count,3),dtype='float64'))


@register_atom(witness_andriy_r_glm_segment_probabilities)
def andriy_r_glm_segment_probabilities(training_features: list,training_labels: list,prediction_features: list,valid_prediction_rows: list) -> tuple[NDArray[np.float64],NDArray[np.float64]]:
    """Fit the source GLM branch for three populations; return ranks/raw scores.

    Require normalized windows/1965 feature matrices, positive-first binary
    training labels and boolean prediction masks, in lists of three populations.
    Training is finite; prediction NaNs become zero. A 1500-round XGBoost
    selector must use at least 200 features. Explicit one-based feature names
    adapt current R importance output. Fit source binomial glmnet alpha0 paths
    on top200/top100 columns, predicting at s=.0001 with native path interpolation
    or boundary clamping. Separately use training means/sample SD and LiblineaR
    type6, cost10, bias TRUE on top200; verify first probability column is class1.

    Maximize all 19 window probabilities per segment, including invalid windows;
    zero only wholly invalid segments, then average normalized per-model ranks
    across all populations. Return rank scores and segments/3 pre-rank scores.
    Source seed1234 precedes the population loop. Require R4.6.1, XGBoost3.2.1.1,
    glmnet5.0, LiblineaR2.10.24 and R.matlab3.8.0; SCIONA_RSCRIPT/R_LIBS_USER select
    the runtime. Current defaults/serial execution; no historical runtime claim.
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
        raise RuntimeError('Rscript is required for the Andriy R GLM provider')
    with TemporaryDirectory(prefix='sciona-andriy-r-glm-') as temporary:
        directory=Path(temporary)
        savemat(directory/'input.mat',payload)
        (directory/'run.R').write_text(_R_SCRIPT)
        try:
            run=subprocess.run([executable,'--vanilla',str(directory/'run.R'),str(directory/'input.mat'),str(directory/'output.mat')],
                env=dict(os.environ,OMP_NUM_THREADS='1'),capture_output=True,timeout=1200)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError('Andriy R GLM execution timed out') from exc
        if run.returncode:
            raise RuntimeError('Andriy R GLM execution failed; check the pinned R runtime and packages')
        output=loadmat(directory/'output.mat')
    scores=output['scores'].reshape(-1)
    segments=output['segments']
    if scores.shape!=(total_segments,) or segments.shape!=(total_segments,3) or not np.all(np.isfinite(scores)) or not np.all(np.isfinite(segments)):
        raise RuntimeError('invalid Andriy R GLM output')
    return scores,segments
