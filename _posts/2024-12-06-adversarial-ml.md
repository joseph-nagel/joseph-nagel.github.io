---
layout: post
title: Adversarial ML
mathjax: true
tags: [Introduction, Adversarial ML, Adversarial attacks]
thumbnail-img: https://raw.githubusercontent.com/joseph-nagel/adversarial-ml/main/assets/attacked.png
gh-repo: joseph-nagel/adversarial-ml
gh-badge: [star, fork, follow]
---

The existence of adversarial examples for neural networks has been first observed in the context of image classification [[Szegedy et al., 2014](https://arxiv.org/abs/1312.6199)]. There are many great review papers on adversarial attacks and corresponding defenses. For example, the following publications are open access: [[Ren et al., 2020](https://doi.org/10.1016/j.eng.2019.12.012); [Khamaiseh et al., 2022](https://doi.org/10.1109/ACCESS.2022.3208131); [Meyers et al., 2023](https://doi.org/10.1007/s10462-023-10521-4); [Liu et al., 2024](https://doi.org/10.1007/s10462-024-10841-z)].


## Adversarial attacks

We consider image classification as a prototypical problem for the occurrence of adversarial inputs. Given a dataset $$\{(\boldsymbol{x}_i, y_i)\}_{i=1}^N$$ of images $$\boldsymbol{x}_i$$ and labels $$y_i$$. The weights $$\boldsymbol{\theta}$$ of a neural network $$\mathcal{M}_{\boldsymbol{\theta}}(\boldsymbol{x})$$ can be found by minimizing a loss function:

$$
\hat{\boldsymbol{\theta}} =
\operatorname*{arg\,min}_{\boldsymbol{\theta}}
\frac{1}{N} \sum_{i=1}^N L(\mathcal{M}_{\boldsymbol{\theta}}(\boldsymbol{x}_i), y_i).
$$

Here, $$L(\mathcal{M}_{\boldsymbol{\theta}}(\boldsymbol{x}), y)$$ is the contribution of a single data point $$(\boldsymbol{x}, y)$$.

Given a trained classifier, one can try to (imperceptibly) perturb an input $$\boldsymbol{x}$$ such the altered image $$\tilde{\boldsymbol{x}} \in \mathcal{B}_p(\boldsymbol{x}, \epsilon)$$ is misclassified. A small $$\ell_p$$-ball $$\mathcal{B}_p(\boldsymbol{x}, \epsilon) = \{\boldsymbol{x}^\star \colon \lVert \boldsymbol{x}^\star -\boldsymbol{x} \rVert_p \leq \epsilon\}$$ with radius $$\epsilon > 0$$ is often used to constrain the image modification. This **adversarial attack** can be formulated as the constrained optimization problem of maximizing the loss:

$$
\tilde{\boldsymbol{x}} =
\operatorname*{arg\,max}_{\boldsymbol{x}^\star \in \mathcal{B}_p(\boldsymbol{x}, \epsilon)}
L(\mathcal{M}_{\boldsymbol{\theta}}(\boldsymbol{x}^\star), y).
$$

Beyond perturbations contained in a small $$\epsilon$$-neighborhood, any image modification that can be reasonably assumed not to change the true class label is admissible here. It is noted that this opens up vast spaces of possible attacks. The attack is successful if $$\mathcal{M}_{\boldsymbol{\theta}}(\tilde{\boldsymbol{x}})$$ predicts the wrong label.

Since the predicted probability of the true class is minimized, without specifying a certain wrong target class, the attack above is called **untargeted**. One may similarly trick the model into predicting a specific label $$\tilde{y}$$ with $$\tilde{y} \neq y$$. Such a **targeted attack** can be formulated as:

$$
\tilde{\boldsymbol{x}} =
\operatorname*{arg\,min}_{\boldsymbol{x}^\star \in \mathcal{B}_p(\boldsymbol{x}, \epsilon)}
L(\mathcal{M}_{\boldsymbol{\theta}}(\boldsymbol{x}^\star), \tilde{y}).
$$


## Gradient-based attacks

The **fast gradient-sign method** (FGSM) was proposed in [[Goodfellow et al., 2015](https://arxiv.org/abs/1412.6572)]. It is a simple and efficient technique to compute adversarial attacks. A perturbation $$\tilde{\boldsymbol{x}}$$ of an input image $$\boldsymbol{x}$$ is computed as

$$
\tilde{\boldsymbol{x}} = \boldsymbol{x} + \epsilon \cdot \operatorname{sign}
\left( \nabla_{\boldsymbol{x}} L(\mathcal{M}_{\boldsymbol{\theta}}(\boldsymbol{x}), y) \right).
$$

This approach can be seen as a first-order order or single-step approximation of the untargeted $$\ell_{\infty}$$-norm attack with the constraint $$\lVert \tilde{\boldsymbol{x}} - \boldsymbol{x} \rVert_{\infty} \leq \epsilon$$. The corresponding targeted variant of the FGSM attack is simply $$\tilde{\boldsymbol{x}} = \boldsymbol{x} - \epsilon \cdot \operatorname{sign}(\nabla_{\boldsymbol{x}} L(\mathcal{M}_{\boldsymbol{\theta}}(\boldsymbol{x}), \tilde{y}))$$.

A straightforward multi-step procedure is the **projected gradient descent** (PGD) attack from [[Carlini and Wagner, 2017](https://doi.org/10.1109/SP.2017.49)]. Starting from $$\boldsymbol{x}$$, or from a random location from its $$\epsilon$$-neighborhood (any $$p$$-norm), it proceeds by iteratively performing a certain number of update steps

$$
\tilde{\boldsymbol{x}}_{t + 1} = \operatorname{proj}_{\epsilon} \left( \tilde{\boldsymbol{x}}_t +
\gamma \cdot \nabla_{\boldsymbol{x}} L(\mathcal{M}_{\boldsymbol{\theta}}(\tilde{\boldsymbol{x}}_t), y) \right).
$$

Here, $$\gamma > 0$$ is a step size parameter, and $$\operatorname{proj}_{\epsilon}$$ denotes the operation of projecting a point outside of the $$\epsilon$$-neighborhood around $$\boldsymbol{x}$$ to the closest point on the surface. While the above describes an untargeted attack, the corresponding targeted approach can be pursued analogously.


## Adversarial training

Many defenses against adversarial attacks have been proposed in the literature. An intuitive approach to encourage models to be robust with respect to certain attacks is **adversarial training** [[Goodfellow et al., 2015](https://arxiv.org/abs/1412.6572); [Madry et al., 2018](https://openreview.net/forum?id=rJzIBfZAb)]. Instead of minimizing the standard loss function, one considers a worst-case formulation:

$$
\hat{\boldsymbol{\theta}} =
\operatorname*{arg\,min}_{\boldsymbol{\theta}} \frac{1}{N} \sum_{i=1}^N
\operatorname*{max}_{\boldsymbol{x}_i^\star \in \mathcal{B}_p(\boldsymbol{x}_i, \epsilon)}
L(\mathcal{M}_{\boldsymbol{\theta}}(\boldsymbol{x}_i^\star), y_i).
$$

The inner optimization of this min-max problem can be addressed by finding specific adversarial examples $$\tilde{\boldsymbol{x}}_i \in \mathcal{B}_p(\boldsymbol{x}_i, \epsilon)$$ with one of the gradient-driven attacks discussed before. The worst-case loss is then under-approximated as $$L(\mathcal{M}_{\boldsymbol{\theta}}(\tilde{\boldsymbol{x}}_i), y_i)$$. Note that this method can be seen as a training-time injection of adversarial examples.

An extension of this idea is to employ the worst-case term only as an adversarial regularization in addition to the usual objective. In this case, a parameter $$\alpha \in (0, 1)$$ can be used in order to weight both loss terms relative to each other:

$$
\hat{\boldsymbol{\theta}} =
\operatorname*{arg\,min}_{\boldsymbol{\theta}}
\frac{1}{N} \sum_{i=1}^N  \left( \alpha \cdot L(\mathcal{M}_{\boldsymbol{\theta}}(\boldsymbol{x}_i), y_i) +
(1 - \alpha) \cdot \operatorname*{max}_{\boldsymbol{x}_i^\star \in \mathcal{B}_p(\boldsymbol{x}_i, \epsilon)}
L(\mathcal{M}_{\boldsymbol{\theta}}(\boldsymbol{x}_i^\star), y_i) \right).
$$
