---
layout: post
title: "Conformal prediction"
mathjax: true
tags: ["Introduction", "Uncertainty quantification", "Conformal prediction", "Prediction sets"]
thumbnail-img: https://raw.githubusercontent.com/joseph-nagel/conformal-prediction/main/assets/classif_aps.jpg
gh-repo: joseph-nagel/conformal-prediction
gh-badge: [star, fork, follow]
---

This blog post contains a brief introduction to **conformal prediction** (CP), a statistical framework for uncertainty quantification in machine learning [[Vovk et al., 2022](https://doi.org/10.1007/978-3-031-06649-8)]. The uncertainty of model predictions is quantified through prediction sets or intervals that contain the true outcome with a high probability. CP has a number of compelling advantages:
- **finite-sample guarantees**: does not need asymptotic assumptions
- **distribution-free**: requires only minimal assumptions on the data distribution
- **model-agnostic**: works for all (pretrained) black-box models

Our discussion here concentrates on the bare minimum of theoretical results and implementational aspects that is necessary in order to appreciate and apply CP. More exhaustive and rigorous overviews can be found in the references [[Tibshirani, 2023](https://www.stat.berkeley.edu/~ryantibs/statlearn-s23/lectures/conformal.pdf); [Da Veiga, 2024](https://hal.science/hal-04690218); [Angelopoulos et al., 2024](https://arxiv.org/abs/2411.11824); [Sesia and Favaro, 2026](https://arxiv.org/abs/2603.23923)].


## Prediction sets

Let us consider a supervised learning task to predict a target variable $$Y \in \mathcal{Y}$$ from a feature vector $$\boldsymbol{X} \in \mathcal{X} \subseteq \mathbb{R}^d$$. The target domain might be $$\mathcal{Y} = \{c_1, \ldots, c_m\}$$ for classification or $$\mathcal{Y} = \mathbb{R}$$ for a regression problem. As usual, one typically assumes a dataset $$D_n = \{(\boldsymbol{X}_i, Y_i)\}_{i=1}^n$$ of i.i.d. feature and target pairs with $$(\boldsymbol{X}_i, Y_i) \sim P$$ for $$i=1,\ldots,n$$.

In this situation, for a new i.i.d. test point $$(\boldsymbol{X}_{n+1}, Y_{n+1}) \sim P$$, the goal of conformal prediction is to construct a **prediction set** or **interval** $$C_n(\boldsymbol{X}_{n + 1}) \subseteq \mathcal{Y}$$ that contains the true outcome $$Y_{n + 1}$$ with at least a probability $$1 - \alpha$$. This property is also called **marginal coverage**:

$$
\mathbb{P}(Y_{n + 1} \in C_n(\boldsymbol{X}_{n + 1})) \geq 1 - \alpha.
$$

Here, $$\alpha \in (0, 1)$$ is a user-defined significance level or error rate. The probability is over all data with $$i = 1,\ldots,n+1$$, which means that predictive coverage is only considered on average over the train and new test data.

Note that the full set $$C_n(\boldsymbol{X}_{n + 1}) = \mathcal{Y}$$ trivially guarantees the predictive coverage property (probability is one). Similarly, a prediction set that satisfies marginal coverage (with equality) could also be specified as

$$
C_n(\boldsymbol{X}_{n + 1}) =
\begin{cases}
\mathcal{Y} & \text{with probability } 1 - \alpha, \\
\emptyset & \text{with probability } \alpha.
\end{cases}
$$

Since such trivial prediction sets are not at all useful in practice, CP is concerned with finding smaller and more informative sets that still satisfy the marginal coverage property.

It is also worth noting that marginal coverage does not imply any conditional coverage guarantee, e.g. conditioned on $$D_n$$, $$\boldsymbol{X}_{n + 1}$$ or $$Y_{n + 1}$$. **Feature-conditional coverage** $$\mathbb{P}(Y_{n + 1} \in C_n(\boldsymbol{X}_{n + 1}) \,\vert\, \boldsymbol{X}_{n + 1} = \boldsymbol{x}) \geq 1 - \alpha$$ for all $$\boldsymbol{x} \in \mathcal{X}$$ is a stronger notion than the marginal variant discussed above. Contrary to marginal coverage, however, it turns out that conditional coverage cannot be achieved for reasonably small prediction sets and large/infinite feature spaces.

Also, the **training-conditional coverage** property $$\mathbb{P}(Y_{n + 1} \in C_n(\boldsymbol{X}_{n + 1}) \,\vert\, D_n) \geq 1 - \alpha$$ would provide a guarantee that holds for any given dataset $$D_n$$, not just on average over the distribution of datasets. The CP technique discussed below features such a kind of training-conditional statistical guarantee.


## Split conformal prediction

We now discuss a surprisingly simple CP procedure that is referred to as **split conformal prediction** (SCP). It is based on a pre-existing point predictor $$\hat{f} \colon \mathcal{X} \mapsto \mathcal{Y}$$. The data that has been used for training this model, denoted as $$D_{\mathrm{pre}}$$, is called the **proper training set** in this context. A disjoint dataset $$D_n = \{(\boldsymbol{X}_i, Y_i)\}_{i=1}^n$$, which is called the **calibration set**, can then be used for constructing the prediction sets.

A **non-conformity score** $$s(\boldsymbol{X}, Y)$$ is a function that measures how different or strange a new example $$(\boldsymbol{X}, Y)$$ is with respect to a reference set. The higher the score, the more atypical the test sample. Score functions may be defined based on the disagreement between the true target outcome $$Y$$ and the corresponding model prediction $$\hat{f}(\boldsymbol{X})$$. For instance, the absolute residual $$s(\boldsymbol{X}, Y) = \lvert Y - \hat{f}(\boldsymbol{X}) \rvert$$ is an appropriate score for regression. For a classification problem, one can use $$s(\boldsymbol{X}, Y) = 1 - \hat{p}(Y \vert \boldsymbol{X})$$, i.e. one minus the (softmax) probability of the true outcome.

One can now compute the set of all non-conformity scores $$\{S_i = s(\boldsymbol{X}_i, Y_i)\}_{i=1}^n$$ and determine $$q_n = \operatorname{Quantile}_{(1 - \alpha)(n + 1) / n}(S_1, \ldots, S_n)$$ as the $$\lceil (1 - \alpha)(n + 1) \rceil$$-th smallest value. Essentially, this is the empirical $$(1 - \alpha)$$-quantile with a small modification that can be seen as a finite-sample correction. The SCP prediction set is then defined as

$$
C_n(\boldsymbol{X}_{n + 1}) = \{y \in \mathcal{Y} : s(\boldsymbol{X}_{n + 1}, y) \leq q_n\}.
$$

Under generally mild but occasionally subtle assumptions (i.i.d. data and the weaker notion of exchangeability play a prominent role here), the SCP prediction set can be shown to satisfy various related coverage guarantees. To begin with, one finds that $$\mathbb{P}(Y_{n + 1} \in C_n(\boldsymbol{X}_{n + 1}) \,\vert\, D_{\mathrm{pre}} \cup D_n)$$ is a beta-distributed random variable that for increasing calibration set size $$n$$ concentrates more and more around $$1 - \alpha$$. Moreover, one has

$$
1 - \alpha \leq \mathbb{P}(Y_{n + 1} \in C_n(\boldsymbol{X}_{n + 1}) \,\vert\, D_{\mathrm{pre}}) \leq 1 - \alpha + \frac{1}{n + 1},
$$

from which also the unconditional property $$\mathbb{P}(Y_{n + 1} \in C_n(\boldsymbol{X}_{n + 1})) = \mathbb{E} \left[ \mathbb{P}(Y_{n + 1} \in C_n(\boldsymbol{X}_{n + 1}) \,\vert\, D_{\mathrm{pre}}) \right] \geq 1 - \alpha$$ follows. The gist of these results is that SCP provides both marginal and training-conditional guarantees.

Interestingly, the SCP procedure works very generally, no matter the data distribution, the pre-fitted model or the conformal score function. But, while the coverage guarantees hold even for bad models, e.g. due to misspecification or overfitting, the size of the prediction sets indeed depends on the quality of the model and the adequateness of the score. Better models tend to produce tighter and therefore more informative prediction sets.

It is noted that the SCP scheme discussed above can be seen as a special case of the more general **full conformal prediction**. The fully conformal procedure is, generally speaking, much more complicated and expensive. It utilizes all data for training and calibration at the cost of repeatedly retraining the model. This way of proceeding is more data-efficient but less computationally efficient.


## Regression scores

In the regression setting, it can be easily seen that SCP in conjunction with the residual score $$s(\boldsymbol{X}, Y) = \lvert Y - \hat{f}(\boldsymbol{X}) \rvert$$ brings about prediction intervals of the form $$C_n(\boldsymbol{X}_{n + 1}) = \hat{f}(\boldsymbol{X}_{n + 1}) \pm q_n$$, i.e. they have the same width across different values of the features. Such constant-width uncertainty bands completely lack local adaptivity to noise, problem hardness and data scarcity. In the following, we will have a short look into how to establish a type of adaptivity. A more thorough review is provided by [[Kato et al., 2023](https://proceedings.mlr.press/v204/kato23a.html)].

A quick fix to enable localized uncertainty quantification is to learn an $$\boldsymbol{X}$$-dependent standard deviation $$\hat{\sigma}(\boldsymbol{X})$$ as an error estimate for the model $$\hat{f}(\boldsymbol{X})$$. Plugging this uncertainty estimate into the definition of the **scaled residual score**

$$
s(\boldsymbol{X}, Y) = \frac{\lvert Y - \hat{f}(\boldsymbol{X}) \rvert}{\hat{\sigma}(\boldsymbol{X})}
$$

directly results in adaptive (but still symmetric) SCP prediction intervals $$C_n(\boldsymbol{X}_{n + 1}) = \hat{f}(\boldsymbol{X}_{n + 1}) \pm \hat{\sigma}(\boldsymbol{X}_{n + 1}) \cdot q_n$$. This simply inherits the adaptivity from the heuristic notion of uncertainty provided by $$\hat{\sigma}(\boldsymbol{X})$$ and multiplicatively corrects it so as to ensure formal statistical guarantees.

One may sometimes want to go beyond symmetric uncertainty estimates. A common approach in this case is **conformalized quantile regression** (CQR). It is grounded on quantile regression to learn $$\hat{f}_{\alpha/2}(\boldsymbol{X})$$ and $$\hat{f}_{1-\alpha/2}(\boldsymbol{X})$$ as the $$\boldsymbol{X}$$-conditional response quantiles of level $$\alpha/2$$ and $$1-\alpha/2$$, respectively. The **CQR score**

$$
s(\boldsymbol{X}, Y) = \operatorname{max} \left( \hat{f}_{\alpha/2}(\boldsymbol{X}) - Y, Y - \hat{f}_{1-\alpha/2}(\boldsymbol{X}) \right)
$$

yields the non-symmetric CQR prediction intervals $$C_n(\boldsymbol{X}_{n + 1}) = [\hat{f}_{\alpha/2}(\boldsymbol{X}_{n + 1}) - q_n, \hat{f}_{1-\alpha/2}(\boldsymbol{X}_{n + 1}) + q_n]$$. They involve an additive correction to the naive interval estimate deriving directly from the learned quantiles. This correction either expands or shrinks the original interval and equips it with the usual SCP guarantees.


## Classification scores

We have already encountered the standard classification score $$s(\boldsymbol{X}, Y) = 1 - \hat{p}(Y \vert \boldsymbol{X})$$ with the associated prediction sets $$C_n(\boldsymbol{X}_{n + 1}) = \{y \in \mathcal{Y} : \hat{p}(y \vert \boldsymbol{X}_{n + 1}) \geq 1 - q_n\}$$. Under the premise that $$\hat{p}(Y \vert \boldsymbol{X})$$ is perfect, this method is known to produce the smallest prediction sets on average [[Sadinle et al., 2019](https://arxiv.org/abs/1609.00451); [Dhillon et al., 2024](https://proceedings.mlr.press/v238/dhillon24a.html)]. Hence, it is sometimes referred to as **least ambiguous set-valued classifier** (LAC). Beyond the oracle scenario, the optimality may still apply in some asymptotic sense for imperfect models. On the downside, the approach falls short in terms of adaptivity and might produce empty sets close to the decision boundary.

A compellingly simple construction of prediction sets proceeds as follows: Sort the class labels from highest to lowest predicted probability and successively include the most probable classes until their accumulated probability mass exceeds $$1 - \alpha$$. Of course, this naive approach would actually come with the desirable guarantees in the hypothetical perfect-model case. In practice, when models are imperfect, it can still be seen as a heuristic.

The naive strategy discussed in the last paragraph has inspired the idea of **adaptive prediction sets** (APS) [[Romano et al., 2020](https://arxiv.org/abs/2006.02544)]. Here, the cumulative probability of all classes that are at least as likely as the true one is taken as a scoring function

$$
s^{\mathrm{APS}}(\boldsymbol{X}, Y) = \sum_{y \in \mathcal{Y}} \hat{p}(y \vert \boldsymbol{X}) \, \boldsymbol{1}_{\hat{p}(y \vert \boldsymbol{X}) \geq \hat{p}(Y \vert \boldsymbol{X})}.
$$

In comparison to LAC, this score involves softmax outputs from more than just a single class. The SCP prediction sets $$C_n(\boldsymbol{X}_{n + 1}) = \{y \in \mathcal{Y} : s^{\mathrm{APS}}(\boldsymbol{X}_{n + 1}, y) \leq q_n\}$$ need to be slightly modified for APS. This is because, in the event that $$\operatorname{max}_{y \in \mathcal{Y}} \hat{p}(y \vert \boldsymbol{X}_{n + 1}) > q_n$$, even the top-probability class would be excluded. In order to prevent this from happening, the APS prediction sets always explicitly include the top-ranked class:

$$
C^{\mathrm{APS}}_n(\boldsymbol{X}_{n + 1}) = \left\{ y \in \mathcal{Y} : s^{\mathrm{APS}}(\boldsymbol{X}_{n + 1}, y) \leq q_n \right\} \cup
\operatorname*{arg\,max}_{y \in \mathcal{Y}} \hat{p}(y \vert \boldsymbol{X}_{n + 1}).
$$

Actually, it is not so straightforward to interpret $$s^{\mathrm{APS}}(\boldsymbol{X}, Y)$$ as a non-conformity measure. This is exemplified by observing that both edge cases $$\hat{p}(Y \vert \boldsymbol{X}) = 1$$ and $$\hat{p}(Y \vert \boldsymbol{X}) = 0$$ would lead to the high-score $$s^{\mathrm{APS}}(\boldsymbol{X}, Y) = 1$$. This and a number of other issues are addressed by **regularized APS** (RAPS) [[Angelopoulos et al., 2021](https://openreview.net/forum?id=eNdiU_DbM9)]. A regularization parameter penalizing a larger number of contributing classes is introduced in the score

$$
s^{\mathrm{RAPS}}(\boldsymbol{X}, Y) = s^{\mathrm{APS}}(\boldsymbol{X}, Y) + \lambda \cdot \operatorname{max}(0, k(\boldsymbol{X}, Y) - k_{\mathrm{reg}}).
$$

Here, $$k(\boldsymbol{X}, Y) = \sum_{y \in \mathcal{Y}} \boldsymbol{1}_{\hat{p}(y \vert \boldsymbol{X}) \geq \hat{p}(Y \vert \boldsymbol{X})}$$ is the number of classes that contribute to $$s^{\mathrm{APS}}(\boldsymbol{X}, Y)$$, $$k_{\mathrm{reg}}$$ is the maximum number of classes that is allowed without incurring a penalty, and $$\lambda$$ is the regularization weight. The RAPS score encourages small sets and avoids the effect of noise in the tails of $$\hat{p}(\cdot \vert \boldsymbol{X})$$.
