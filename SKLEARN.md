# scikit-learn Ingestion Targets

This file is the upstream target inventory for sklearn ingestion.

Execution sequencing, restart-safe handoff notes, difficulty tiers, and the
recommended first-wave targets now live in
[SKLEARN_EXECUTION_PLAN.md](/Users/conrad/personal/ageo-atoms/SKLEARN_EXECUTION_PLAN.md).

This inventory is based on the official stable API reference for scikit-learn 1.8.0.

- API reference: https://scikit-learn.org/stable/api/index.html
- Scope for this first ingestion pass: public algorithmic estimators, transformers, kernels, and algorithmic helper functions in the selected sklearn modules below.
- Intentionally excluded for now: datasets, metrics, displays, exceptions, low-level utils, orchestration/composition (`compose`, `pipeline`), and model-selection/search APIs.
- Source locations come from each object page's official `[source]` link. `BallTree` and `KDTree` use the matching source files from the same scikit-learn revision because their generated docs pages do not expose a direct `[source]` link.
- Total targets in this inventory: 108

## Module Summary

| Module | Targets |
| --- | ---: |
| `sklearn.calibration` | 0 |
| `sklearn.cluster` | 0 |
| `sklearn.covariance` | 0 |
| `sklearn.cross_decomposition` | 0 |
| `sklearn.decomposition` | 14 |
| `sklearn.discriminant_analysis` | 0 |
| `sklearn.dummy` | 0 |
| `sklearn.ensemble` | 18 |
| `sklearn.feature_extraction` | 0 |
| `sklearn.feature_extraction.image` | 0 |
| `sklearn.feature_extraction.text` | 1 |
| `sklearn.feature_selection` | 6 |
| `sklearn.gaussian_process` | 0 |
| `sklearn.gaussian_process.kernels` | 0 |
| `sklearn.impute` | 0 |
| `sklearn.inspection` | 0 |
| `sklearn.kernel_approximation` | 0 |
| `sklearn.kernel_ridge` | 0 |
| `sklearn.linear_model` | 34 |
| `sklearn.manifold` | 8 |
| `sklearn.mixture` | 2 |
| `sklearn.multiclass` | 3 |
| `sklearn.multioutput` | 4 |
| `sklearn.naive_bayes` | 0 |
| `sklearn.neighbors` | 15 |
| `sklearn.neural_network` | 3 |
| `sklearn.preprocessing` | 0 |
| `sklearn.semi_supervised` | 0 |
| `sklearn.svm` | 0 |
| `sklearn.tree` | 0 |

## `sklearn.calibration`

| Target | Description | Source |
| --- | --- | --- |

## `sklearn.cluster`

| Target | Description | Source |
| --- | --- | --- |

## `sklearn.covariance`

| Target | Description | Source |
| --- | --- | --- |

## `sklearn.cross_decomposition`

| Target | Description | Source |
| --- | --- | --- |

## `sklearn.decomposition`

| Target | Description | Source |
| --- | --- | --- |
| `dict_learning` | Solve a dictionary learning matrix factorization problem. | `sklearn/decomposition/_dict_learning.py:L883` |
| `dict_learning_online` | Solve a dictionary learning matrix factorization problem online. | `sklearn/decomposition/_dict_learning.py:L664` |
| `DictionaryLearning` | Dictionary learning. | `sklearn/decomposition/_dict_learning.py:L1417` |
| `FastICA` | FastICA: a fast algorithm for Independent Component Analysis. | `sklearn/decomposition/_fastica.py:L372` |
| `fastica` | Perform Fast Independent Component Analysis. | `sklearn/decomposition/_fastica.py:L168` |
| `LatentDirichletAllocation` | Latent Dirichlet Allocation with online variational Bayes algorithm. | `sklearn/decomposition/_lda.py:L160` |
| `MiniBatchDictionaryLearning` | Mini-batch dictionary learning. | `sklearn/decomposition/_dict_learning.py:L1760` |
| `MiniBatchNMF` | Mini-Batch Non-Negative Matrix Factorization (NMF). | `sklearn/decomposition/_nmf.py:L1758` |
| `MiniBatchSparsePCA` | Mini-batch Sparse Principal Components Analysis. | `sklearn/decomposition/_sparse_pca.py:L342` |
| `NMF` | Non-Negative Matrix Factorization (NMF). | `sklearn/decomposition/_nmf.py:L1317` |
| `non_negative_factorization` | Compute Non-negative Matrix Factorization (NMF). | `sklearn/decomposition/_nmf.py:L888` |
| `sparse_encode` | Sparse coding. | `sklearn/decomposition/_dict_learning.py:L204` |
| `SparseCoder` | Sparse coding. | `sklearn/decomposition/_dict_learning.py:L1182` |
| `SparsePCA` | Sparse Principal Components Analysis (SparsePCA). | `sklearn/decomposition/_sparse_pca.py:L162` |

## `sklearn.discriminant_analysis`

| Target | Description | Source |
| --- | --- | --- |

## `sklearn.dummy`

| Target | Description | Source |
| --- | --- | --- |

## `sklearn.ensemble`

| Target | Description | Source |
| --- | --- | --- |
| `AdaBoostClassifier` | An AdaBoost classifier. | `sklearn/ensemble/_weight_boosting.py:L321` |
| `AdaBoostRegressor` | An AdaBoost regressor. | `sklearn/ensemble/_weight_boosting.py:L823` |
| `BaggingClassifier` | A Bagging classifier. | `sklearn/ensemble/_bagging.py:L741` |
| `BaggingRegressor` | A Bagging regressor. | `sklearn/ensemble/_bagging.py:L1253` |
| `ExtraTreesClassifier` | An extra-trees classifier. | `sklearn/ensemble/_forest.py:L1944` |
| `ExtraTreesRegressor` | An extra-trees regressor. | `sklearn/ensemble/_forest.py:L2328` |
| `GradientBoostingClassifier` | Gradient Boosting for classification. | `sklearn/ensemble/_gb.py:L1134` |
| `GradientBoostingRegressor` | Gradient Boosting for regression. | `sklearn/ensemble/_gb.py:L1746` |
| `HistGradientBoostingClassifier` | Histogram-based Gradient Boosting Classification Tree. | `sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py:L1875` |
| `HistGradientBoostingRegressor` | Histogram-based Gradient Boosting Regression Tree. | `sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py:L1472` |
| `IsolationForest` | Isolation Forest Algorithm. | `sklearn/ensemble/_iforest.py:L55` |
| `RandomForestClassifier` | A random forest classifier. | `sklearn/ensemble/_forest.py:L1174` |
| `RandomForestRegressor` | A random forest regressor. | `sklearn/ensemble/_forest.py:L1572` |
| `RandomTreesEmbedding` | An ensemble of totally random trees. | `sklearn/ensemble/_forest.py:L2679` |
| `StackingClassifier` | Stack of estimators with a final classifier. | `sklearn/ensemble/_stacking.py:L422` |
| `StackingRegressor` | Stack of estimators with a final regressor. | `sklearn/ensemble/_stacking.py:L841` |
| `VotingClassifier` | Soft Voting/Majority Rule classifier for unfitted estimators. | `sklearn/ensemble/_voting.py:L194` |
| `VotingRegressor` | Prediction voting regressor for unfitted estimators. | `sklearn/ensemble/_voting.py:L542` |

## `sklearn.feature_extraction`

| Target | Description | Source |
| --- | --- | --- |

## `sklearn.feature_extraction.image`

| Target | Description | Source |
| --- | --- | --- |

## `sklearn.feature_extraction.text`

| Target | Description | Source |
| --- | --- | --- |
| `TfidfVectorizer` | Convert a collection of raw documents to a matrix of TF-IDF features. | `sklearn/feature_extraction/text.py:L1744` |

## `sklearn.feature_selection`

| Target | Description | Source |
| --- | --- | --- |
| `mutual_info_classif` | Estimate mutual information for a discrete target variable. | `sklearn/feature_selection/_mutual_info.py:L453` |
| `mutual_info_regression` | Estimate mutual information for a continuous target variable. | `sklearn/feature_selection/_mutual_info.py:L325` |
| `RFE` | Feature ranking with recursive feature elimination. | `sklearn/feature_selection/_rfe.py:L73` |
| `RFECV` | Recursive feature elimination with cross-validation to select features. | `sklearn/feature_selection/_rfe.py:L558` |
| `SelectFromModel` | Meta-transformer for selecting features based on importance weights. | `sklearn/feature_selection/_from_model.py:L95` |
| `SequentialFeatureSelector` | Transformer that performs Sequential Feature Selection. | `sklearn/feature_selection/_sequential.py:L34` |

## `sklearn.gaussian_process`

| Target | Description | Source |
| --- | --- | --- |

## `sklearn.gaussian_process.kernels`

| Target | Description | Source |
| --- | --- | --- |

## `sklearn.impute`

| Target | Description | Source |
| --- | --- | --- |

## `sklearn.inspection`

| Target | Description | Source |
| --- | --- | --- |

## `sklearn.kernel_approximation`

| Target | Description | Source |
| --- | --- | --- |

## `sklearn.kernel_ridge`

| Target | Description | Source |
| --- | --- | --- |

## `sklearn.linear_model`

| Target | Description | Source |
| --- | --- | --- |
| `ARDRegression` | Bayesian ARD regression. | `sklearn/linear_model/_bayes.py:L466` |
| `BayesianRidge` | Bayesian ridge regression. | `sklearn/linear_model/_bayes.py:L26` |
| `ElasticNet` | Linear regression with combined L1 and L2 priors as regularizer. | `sklearn/linear_model/_coordinate_descent.py:L758` |
| `ElasticNetCV` | Elastic Net model with iterative fitting along a regularization path. | `sklearn/linear_model/_coordinate_descent.py:L2237` |
| `enet_path` | Compute elastic net path with coordinate descent. | `sklearn/linear_model/_coordinate_descent.py:L393` |
| `GammaRegressor` | Generalized Linear Model with a Gamma distribution. | `sklearn/linear_model/_glm/glm.py:L606` |
| `HuberRegressor` | L2-regularized linear regression model that is robust to outliers. | `sklearn/linear_model/_huber.py:L129` |
| `Lars` | Least Angle Regression model a.k.a. LAR. | `sklearn/linear_model/_least_angle.py:L920` |
| `lars_path` | Compute Least Angle Regression or Lasso path using the LARS algorithm. | `sklearn/linear_model/_least_angle.py:L44` |
| `lars_path_gram` | The lars_path in the sufficient stats mode. | `sklearn/linear_model/_least_angle.py:L235` |
| `LarsCV` | Cross-validated Least Angle Regression model. | `sklearn/linear_model/_least_angle.py:L1515` |
| `Lasso` | Linear Model trained with L1 prior as regularizer (aka the Lasso). | `sklearn/linear_model/_coordinate_descent.py:L1205` |
| `lasso_path` | Compute Lasso path with coordinate descent. | `sklearn/linear_model/_coordinate_descent.py:L199` |
| `LassoCV` | Lasso linear model with iterative fitting along a regularization path. | `sklearn/linear_model/_coordinate_descent.py:L1970` |
| `LassoLars` | Lasso model fit with Least Angle Regression a.k.a. Lars. | `sklearn/linear_model/_least_angle.py:L1210` |
| `LassoLarsCV` | Cross-validated Lasso, using the LARS algorithm. | `sklearn/linear_model/_least_angle.py:L1831` |
| `LassoLarsIC` | Lasso model fit with Lars using BIC or AIC for model selection. | `sklearn/linear_model/_least_angle.py:L2029` |
| `LogisticRegression` | Logistic Regression (aka logit, MaxEnt) classifier. | `sklearn/linear_model/_logistic.py:L735` |
| `LogisticRegressionCV` | Logistic Regression CV (aka logit, MaxEnt) classifier. | `sklearn/linear_model/_logistic.py:L1363` |
| `MultiTaskElasticNet` | Multi-task ElasticNet model trained with L1/L2 mixed-norm as regularizer. | `sklearn/linear_model/_coordinate_descent.py:L2532` |
| `MultiTaskElasticNetCV` | Multi-task L1/L2 ElasticNet with built-in cross-validation. | `sklearn/linear_model/_coordinate_descent.py:L2926` |
| `MultiTaskLasso` | Multi-task Lasso model trained with L1/L2 mixed-norm as regularizer. | `sklearn/linear_model/_coordinate_descent.py:L2784` |
| `MultiTaskLassoCV` | Multi-task Lasso model trained with L1/L2 mixed-norm as regularizer. | `sklearn/linear_model/_coordinate_descent.py:L3195` |
| `PassiveAggressiveClassifier` | Passive Aggressive Classifier. | `sklearn/linear_model/_passive_aggressive.py:L17` |
| `PassiveAggressiveRegressor` | Passive Aggressive Regressor. | `sklearn/linear_model/_passive_aggressive.py:L343` |
| `Perceptron` | Linear perceptron classifier. | `sklearn/linear_model/_perceptron.py:L10` |
| `PoissonRegressor` | Generalized Linear Model with a Poisson distribution. | `sklearn/linear_model/_glm/glm.py:L475` |
| `QuantileRegressor` | Linear regression model that predicts conditional quantiles. | `sklearn/linear_model/_quantile.py:L20` |
| `RANSACRegressor` | RANSAC (RANdom SAmple Consensus) algorithm. | `sklearn/linear_model/_ransac.py:L81` |
| `SGDClassifier` | Linear classifiers (SVM, logistic regression, etc.) with SGD training. | `sklearn/linear_model/_stochastic_gradient.py:L950` |
| `SGDOneClassSVM` | Solves linear One-Class SVM using Stochastic Gradient Descent. | `sklearn/linear_model/_stochastic_gradient.py:L2117` |
| `SGDRegressor` | Linear model fitted by minimizing a regularized empirical loss with SGD. | `sklearn/linear_model/_stochastic_gradient.py:L1794` |
| `TheilSenRegressor` | Theil-Sen Estimator: robust multivariate regression model. | `sklearn/linear_model/_theil_sen.py:L207` |
| `TweedieRegressor` | Generalized Linear Model with a Tweedie distribution. | `sklearn/linear_model/_glm/glm.py:L738` |

## `sklearn.manifold`

| Target | Description | Source |
| --- | --- | --- |
| `Isomap` | Isomap Embedding. | `sklearn/manifold/_isomap.py:L28` |
| `locally_linear_embedding` | Perform a Locally Linear Embedding analysis on the data. | `sklearn/manifold/_locally_linear.py:L447` |
| `LocallyLinearEmbedding` | Locally Linear Embedding. | `sklearn/manifold/_locally_linear.py:L601` |
| `MDS` | Multidimensional scaling. | `sklearn/manifold/_mds.py:L440` |
| `smacof` | Compute multidimensional scaling using the SMACOF algorithm. | `sklearn/manifold/_mds.py:L199` |
| `spectral_embedding` | Project the sample on the first eigenvectors of the graph Laplacian. | `sklearn/manifold/_spectral_embedding.py:L150` |
| `SpectralEmbedding` | Spectral embedding for non-linear dimensionality reduction. | `sklearn/manifold/_spectral_embedding.py:L466` |
| `TSNE` | T-distributed Stochastic Neighbor Embedding. | `sklearn/manifold/_t_sne.py:L560` |

## `sklearn.mixture`

| Target | Description | Source |
| --- | --- | --- |
| `BayesianGaussianMixture` | Variational Bayesian estimation of a Gaussian mixture. | `sklearn/mixture/_bayesian_mixture.py:L74` |
| `GaussianMixture` | Gaussian Mixture. | `sklearn/mixture/_gaussian_mixture.py:L556` |

## `sklearn.multiclass`

| Target | Description | Source |
| --- | --- | --- |
| `OneVsOneClassifier` | One-vs-one multiclass strategy. | `sklearn/multiclass.py:L678` |
| `OneVsRestClassifier` | One-vs-the-rest (OvR) multiclass strategy. | `sklearn/multiclass.py:L202` |
| `OutputCodeClassifier` | (Error-Correcting) Output-Code multiclass strategy. | `sklearn/multiclass.py:L1043` |

## `sklearn.multioutput`

| Target | Description | Source |
| --- | --- | --- |
| `ClassifierChain` | A multi-label model that arranges binary classifiers into a chain. | `sklearn/multioutput.py:L877` |
| `MultiOutputClassifier` | Multi target classification. | `sklearn/multioutput.py:L445` |
| `MultiOutputRegressor` | Multi target regression. | `sklearn/multioutput.py:L342` |
| `RegressorChain` | A multi-label model that arranges regressions into a chain. | `sklearn/multioutput.py:L1167` |

## `sklearn.naive_bayes`

| Target | Description | Source |
| --- | --- | --- |

## `sklearn.neighbors`

| Target | Description | Source |
| --- | --- | --- |
| `BallTree` | BallTree for fast generalized N-point problems | `sklearn/neighbors/_ball_tree.pyx.tp:L282` |
| `KDTree` | KDTree for fast generalized N-point problems | `sklearn/neighbors/_kd_tree.pyx.tp:L334` |
| `KernelDensity` | Kernel Density Estimation. | `sklearn/neighbors/_kde.py:L42` |
| `kneighbors_graph` | Compute the (weighted) graph of k-Neighbors for points in X. | `sklearn/neighbors/_graph.py:L50` |
| `KNeighborsClassifier` | Classifier implementing the k-nearest neighbors vote. | `sklearn/neighbors/_classification.py:L44` |
| `KNeighborsRegressor` | Regression based on k-nearest neighbors. | `sklearn/neighbors/_regression.py:L21` |
| `KNeighborsTransformer` | Transform X into a (weighted) graph of k nearest neighbors. | `sklearn/neighbors/_graph.py:L263` |
| `LocalOutlierFactor` | Unsupervised Outlier Detection using the Local Outlier Factor (LOF). | `sklearn/neighbors/_lof.py:L19` |
| `NearestCentroid` | Nearest centroid classifier. | `sklearn/neighbors/_nearest_centroid.py:L26` |
| `NearestNeighbors` | Unsupervised learner for implementing neighbor searches. | `sklearn/neighbors/_unsupervised.py:L10` |
| `NeighborhoodComponentsAnalysis` | Neighborhood Components Analysis. | `sklearn/neighbors/_nca.py:L34` |
| `radius_neighbors_graph` | Compute the (weighted) graph of Neighbors for points in X. | `sklearn/neighbors/_graph.py:L155` |
| `RadiusNeighborsClassifier` | Classifier implementing a vote among neighbors within a given radius. | `sklearn/neighbors/_classification.py:L459` |
| `RadiusNeighborsRegressor` | Regression based on neighbors within a fixed radius. | `sklearn/neighbors/_regression.py:L275` |
| `RadiusNeighborsTransformer` | Transform X into a (weighted) graph of neighbors nearer than a radius. | `sklearn/neighbors/_graph.py:L489` |

## `sklearn.neural_network`

| Target | Description | Source |
| --- | --- | --- |
| `BernoulliRBM` | Bernoulli Restricted Boltzmann Machine (RBM). | `sklearn/neural_network/_rbm.py:L25` |
| `MLPClassifier` | Multi-layer Perceptron classifier. | `sklearn/neural_network/_multilayer_perceptron.py:L879` |
| `MLPRegressor` | Multi-layer Perceptron regressor. | `sklearn/neural_network/_multilayer_perceptron.py:L1386` |

## `sklearn.preprocessing`

| Target | Description | Source |
| --- | --- | --- |

## `sklearn.semi_supervised`

| Target | Description | Source |
| --- | --- | --- |

## `sklearn.svm`

| Target | Description | Source |
| --- | --- | --- |

## `sklearn.tree`

| Target | Description | Source |
| --- | --- | --- |
