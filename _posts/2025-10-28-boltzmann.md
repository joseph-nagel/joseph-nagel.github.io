---
layout: post
title: "Boltzmann & ML"
mathjax: true
tags: ["Boltzmann distribution", "Generative modeling", "Energy-based models"]
---

There are intriguing connections between machine learning and physics. A full appreciation of those formal relationships and analogies is probably beyond my little (remaining) understanding of physics, and certainly very much beyond the scope of this post. We focus on the Boltzmann distribution in statistical physics, the softmax function and energy-based generative models.


## Boltzmann distribution

In classical statistical mechanics, the **Boltzmann distribution** describes the probability $$p_i$$ of states with discrete energies $$\epsilon_i$$ of certain systems that are in thermal equilibrium with a heat bath. It can be written in natural units, when setting the Boltzmann constant to $$k_B = 1$$, as

$$
p_i = \frac{1}{Z} \exp \left( -\frac{\epsilon_i}{T} \right), \quad
Z = \sum_{j=1}^M \exp \left( -\frac{\epsilon_j}{T} \right).
$$

In a way that depends on the temperature $$T$$, states with lower energy have higher occupancies.
The normalization factor $$Z$$ is called the **partition function**.

The Boltzmann distribution is known to maximize the entropy under the condition that the mean energy $$\sum_{j=1}^M p_j \epsilon_j$$ has a certain value (excluding the minimum and maximum energy).


## Softmax function

The **softmax function** is a common building block of neural networks. For multiclass classification, it is often used as the final operation to transform a vector of real-valued logits $$\boldsymbol{z} = (z_1, \ldots, z_M)$$ into (positive and normalized) probabilities

$$
p_i = \operatorname{Softmax}(\boldsymbol{z})_i = \frac{\exp(z_i)}{\sum_{j=1}^M \exp(z_j)}.
$$

This represents a soft version of the argmax. Since the softmax is smooth and differentiable, it establishes a convenient activation function. Beyond multi-output classifiers, it is found in the attention mechanism of the transformer architecture (see [this post](/2024/12/08/attention.html)).

One can easily see that the Boltzmann distribution can be written in terms of the softmax function as $$(p_1, \ldots, p_M) = \operatorname{Softmax}(-\epsilon_1 / T, \ldots, -\epsilon_M / T)$$. Here, $$z_i = -\epsilon_i / T$$ are interpreted as the logits.

The latter is reminiscent of **temperature scaling** [[Hinton et al., 2015](https://arxiv.org/abs/1503.02531); [Guo et al., 2017](https://proceedings.mlr.press/v70/guo17a.html)], where a fictive temperature $$T > 0$$ is introduced in order to modify the probabilities $$p_i \propto \exp(z_i / T)$$. This sharpens ($$T < 1$$) or smoothens ($$T > 1$$) the distribution without altering the top-class prediction.


## Energy-based models

The above is formally related to **energy-based models** (EBMs) [[Huembeli et al., 2022](https://physicsofebm.github.io/); [Carbone, 2024](https://arxiv.org/abs/2406.13661)]. They are generative models that represent the probability density in the Boltzmann form

$$
q_{\boldsymbol{\theta}}(\boldsymbol{x}) = \frac{\exp(-E_{\boldsymbol{\theta}}(\boldsymbol{x}))}{Z_{\boldsymbol{\theta}}}, \quad
Z_{\boldsymbol{\theta}} = \int \exp(-E_{\boldsymbol{\theta}}(\boldsymbol{x})) \, \mathrm{d}\boldsymbol{x}.
$$

Because exponentiation and normalization already ensure the validity of the distribution, the energy $$E_{\boldsymbol{\theta}}(\boldsymbol{x})$$ can be quite an arbitrary parametrized function. It is noted though that the partition function $$Z_{\boldsymbol{\theta}}$$ involves an intractable integral.

Training EBMs requires specialized methods because the likelihood cannot be directly evaluated. Maximum likelihood estimation is enabled by simulating gradients via **Markov chain Monte Carlo** (MCMC) sampling.

Given a set of points $$\{\boldsymbol{x}_i, \ldots, \boldsymbol{x}_N\}$$ that were independently drawn from the true data distribution $$\boldsymbol{x}_i \sim p(\cdot)$$, the log-likelihood (as a function of the parameters $$\boldsymbol{\theta}$$) is given as $$\log q_{\boldsymbol{\theta}}(\boldsymbol{x}_1, \ldots, \boldsymbol{x}_N) = \frac{1}{N} \sum_{i=1}^N \log q_{\boldsymbol{\theta}}(\boldsymbol{x}_i)$$.

Maximizing the log-likelihood is the empirical equivalent of minimizing the expectation $$\mathbb{E}_p[- \log q_{\boldsymbol{\theta}}(\cdot)] = \mathbb{E}_p[E_{\boldsymbol{\theta}}(\cdot)] + \log Z_{\boldsymbol{\theta}}$$. One can derive for the gradients

$$
\nabla _{\boldsymbol{\theta}} \mathbb{E}_p[- \log q_{\boldsymbol{\theta}}(\cdot)] =
\mathbb{E}_p[\nabla_{\boldsymbol{\theta}} E_{\boldsymbol{\theta}}(\cdot)] -
\mathbb{E}_{q_{\boldsymbol{\theta}}}[\nabla_{\boldsymbol{\theta}} E_{\boldsymbol{\theta}}(\cdot)].
$$

The first term can be estimated with the data samples, whereas the second term requires samples from the model distribution. They can be generated with MCMC, which allows for sampling a distribution given its unnormalized density.

There is a variety of MCMC schemes for training EBMs by maximum likelihood. They need to deal with biased samples due to non-converged Markov chains. Beyond that, there are alternative MCMC-free approaches such as **score matching**. A great overview of training methods is provided in [[Song and Kingma, 2021](https://arxiv.org/abs/2101.03288)].


## Joint EBMs

Let us consider a multiclass problem where a neural network $$\mathcal{M}_{\boldsymbol{\theta}} \colon \mathbb{R}^D \to \mathbb{R}^M$$ maps a vector of inputs $$\boldsymbol{x} \in \mathbb{R}^D$$ to a vector of logits $$\mathcal{M}_{\boldsymbol{\theta}}(\boldsymbol{x}) \in \mathbb{R}^M$$. The categorical distribution of the classes $$y \in \{1, \ldots, M\}$$, for given inputs, follows then from applying the softmax to the logits. For completeness, we write

$$
p_{\boldsymbol{\theta}}(y \vert \boldsymbol{x}) = \operatorname{Softmax}(\mathcal{M}_{\boldsymbol{\theta}}(\boldsymbol{x})) =
\frac{\exp(\mathcal{M}_{\boldsymbol{\theta}}(\boldsymbol{x})_y)}{\sum_{y^\prime} \exp(\mathcal{M}_{\boldsymbol{\theta}}(\boldsymbol{x})_{y^\prime})}.
$$

Here, $$\mathcal{M}_{\boldsymbol{\theta}}(\boldsymbol{x})_y$$ denotes the $$y$$-th component of the logits vector.

An interesting relation between such softmax-based classifiers and EBMs has been highlighted in [[Grathwohl et al., 2019](https://arxiv.org/abs/1912.03263)]. A **joint EBM** of the distribution $$p_{\boldsymbol{\theta}}(y, \boldsymbol{x})$$ is constructed that features the generative distribution $$p_{\boldsymbol{\theta}}(\boldsymbol{x})$$ as its marginal and allows one to obtain the discriminative distribution $$p_{\boldsymbol{\theta}}(y \vert \boldsymbol{x})$$ by conditioning.

The key observation is that one can use the logits to define an energy function $$E_{\boldsymbol{\theta}}(y, \boldsymbol{x}) = - \mathcal{M}_{\boldsymbol{\theta}}(\boldsymbol{x})_y$$.
Accordingly, the joint EBM parametrizes the density of both classes $$y$$ and inputs $$\boldsymbol{x}$$ in the form

$$
p_{\boldsymbol{\theta}}(y, \boldsymbol{x}) = \frac{\exp(\mathcal{M}_{\boldsymbol{\theta}}(\boldsymbol{x})_y)}{Z_{\boldsymbol{\theta}}}, \quad
Z_{\boldsymbol{\theta}} = \sum_{y^\prime=1}^M \int_{\mathbb{R}^D} \exp(\mathcal{M}_{\boldsymbol{\theta}}(\boldsymbol{x^\prime})_{y^\prime}) \, \mathrm{d}\boldsymbol{x^\prime}.
$$

Now, from this joint distribution one can easily obtain the marginal $$p_{\boldsymbol{\theta}}(\boldsymbol{x})$$ of the inputs by summing over the classes

$$
p_{\boldsymbol{\theta}}(\boldsymbol{x}) = \sum_{y^\prime=1}^M p_{\boldsymbol{\theta}}(y^\prime, \boldsymbol{x}) =
\frac{\sum_{y^\prime} \exp(\mathcal{M}_{\boldsymbol{\theta}}(\boldsymbol{x})_{y^\prime})}{Z_{\boldsymbol{\theta}}}.
$$

By looking at the numerator of the last fraction, one can identify the energy function of this distribution as $$E_{\boldsymbol{\theta}}(\boldsymbol{x}) = - \log \sum_{y^\prime} \exp(\mathcal{M}_{\boldsymbol{\theta}}(\boldsymbol{x})_{y^\prime})$$.

The discriminative distribution $$p_{\boldsymbol{\theta}}(y \vert \boldsymbol{x})$$ is then recovered by plugging in the two expressions above into the definition of conditional probability

$$
p_{\boldsymbol{\theta}}(y \vert \boldsymbol{x}) = \frac{p_{\boldsymbol{\theta}}(y, \boldsymbol{x})}{p_{\boldsymbol{\theta}}(\boldsymbol{x})} =
\frac{\exp(\mathcal{M}_{\boldsymbol{\theta}}(\boldsymbol{x})_y)}{\sum_{y^\prime} \exp(\mathcal{M}_{\boldsymbol{\theta}}(\boldsymbol{x})_{y^\prime})}.
$$

Note that the factor $$Z_{\boldsymbol{\theta}}$$ has canceled out. Moreover, the standard softmax form discussed in the beginning of this section has re-emerged.


## More physics

Beyond the above, one can of course find more physical concepts in deep learning. See [[Mehta et al., 2019](https://doi.org/10.1016/j.physrep.2019.03.001); [Bahri et al., 2020](https://doi.org/10.1146/annurev-conmatphys-031119-050745); [Decelle, 2023](https://doi.org/10.1016/j.physa.2022.128154)] for review papers from a physics perspective. We conclude with a non-exhaustive list of relevant topics:
- Bayesian statistics, MCMC sampling and variational inference [[Faulkner and Livingstone, 2022](https://arxiv.org/abs/2208.04751); [Grant et al., 2024](https://doi.org/10.1088/1742-5468/ad3350)]
- Symmetries in CNN architectures [[Cohen and Welling, 2016](https://arxiv.org/abs/1602.07576); [Cheng et al., 2019](https://arxiv.org/abs/1906.02481)]
- Physics-informed ML [[Karniadakis et al., 2021](https://doi.org/10.1038/s42254-021-00314-5); [Blechschmidt and Ernst, 2021](https://doi.org/10.1002/gamm.202100006); [Cuomo et al., 2022](https://doi.org/10.1007/s10915-022-01939-z)] (see also [this post](/2024/12/09/pinn.html))
- Dynamical systems and control theory perspective of neural nets [[E et al., 2017](https://doi.org/10.1007/s40304-017-0103-z); [Chen et al., 2018](https://arxiv.org/abs/1806.07366); [Benning et al., 2019](https://arxiv.org/abs/1904.05657); [Li et al., 2023](https://doi.org/10.4171/jems/1221)]
