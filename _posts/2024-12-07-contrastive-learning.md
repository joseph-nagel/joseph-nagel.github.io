---
layout: post
title: Contrastive learning
mathjax: true
tags: [Introduction, Contrastive learning, Triplet loss]
thumbnail-img: https://raw.githubusercontent.com/joseph-nagel/contrastive-learning/main/assets/mnist_embeddings.jpg
gh-repo: joseph-nagel/contrastive-learning
gh-badge: [star, fork, follow]
---

This blog post provides an overly brief introduction to **contrastive representation learning**. In particular, we restrict the discussion to the **contrastive loss** [[Hadsell et al., 2006](https://doi.org/10.1109/CVPR.2006.100)] and the **triplet loss** [[Schroff et al., 2015](https://arxiv.org/abs/1503.03832)]. A more comprehensive review can be found in [[Le-Khac et al., 2020](https://doi.org/10.1109/ACCESS.2020.3031549)]. A common field of application is the problem of **face recognition** [[Wang and Deng, 2018](https://arxiv.org/abs/1804.06655), [Guo and Zhang, 2019](https://doi.org/10.1016/j.cviu.2019.102805)].


## Contrastive loss

The **contrastive loss** is a classic loss function for deep metric learning. For a pair of inputs $$\boldsymbol{x}_i$$ and $$\boldsymbol{x}_j$$, that can be either "similar" to each other or "dissimilar", it can be written as

$$
L(\boldsymbol{x}_i, \boldsymbol{x}_j, y) =
y \cdot \lVert f(\boldsymbol{x}_i) - f(\boldsymbol{x}_j) \rVert_2^2 +
(1 - y) \cdot \operatorname{max} \left( 0, m - \lVert f(\boldsymbol{x}_i) - f(\boldsymbol{x}_j) \rVert_2 \right)^2.
$$

Here, $$y \in \{0,1\}$$ is an indicator variable. A value of $$y=1$$ signifies that the inputs are similar, whereas $$y=0$$ means that they are dissimilar. The encodings $$f(\boldsymbol{x}_i)$$ and $$f(\boldsymbol{x}_j)$$ for similar inputs are consequentially attracted to each other, while they are pushed further apart for dissimilar ones. The margin $$m > 0$$ limits how far the negative samples are repelled.


## Triplet loss

A variation of the loss function above is established by the **triplet loss**. It tries to pull a given **anchor** $$\boldsymbol{a}$$ and a **positive** example $$\boldsymbol{p}$$ closer together, while at the same time pushing the anchor and a **negative** example $$\boldsymbol{n}$$ further away. This is accomplished by the loss

$$
L(\boldsymbol{a}, \boldsymbol{p}, \boldsymbol{n}) =
\operatorname{max} \left( 0, \lVert f(\boldsymbol{a}) - f(\boldsymbol{p}) \rVert_2^2 - \lVert f(\boldsymbol{a}) - f(\boldsymbol{n}) \rVert_2^2 + m \right).
$$

Minimizing this loss tries to push the negative examples more than the margin further away from the anchor than the positive samples. Note that a loss value of zero, as opposed to the contrastive loss above, does not require the positve example to collapse to the anchor.


## Triplet mining

So-called **easy triplets** have zero loss because $$\lVert f(\boldsymbol{a}) - f(\boldsymbol{p}) \rVert_2^2 + m < \lVert f(\boldsymbol{a}) - f(\boldsymbol{n}) \rVert_2^2$$. The negative example is more than the margin further away from the anchor than the positive example is. Triplets where the anchor is closer to the negative than to the positive example, which means $$\lVert f(\boldsymbol{a}) - f(\boldsymbol{n}) \rVert_2^2 < \lVert f(\boldsymbol{a}) - f(\boldsymbol{p}) \rVert_2^2$$, are referred to as  **hard triplets**. For **semi-hard triplets** the positive is closer to the anchor, but not more than the margin. This can be formalized as $$\lVert f(\boldsymbol{a}) - f(\boldsymbol{p}) \rVert_2^2 < \lVert f(\boldsymbol{a}) - f(\boldsymbol{n}) \rVert_2^2 < \lVert f(\boldsymbol{a}) - f(\boldsymbol{p}) \rVert_2^2 + m$$.

**Triplet mining** is the process of selecting triplets for model training. **Offline** strategies choose the corresponding triplets before each training epoch, wheareas **online** mining selects triplets within each batch. The former is expensive since it requires a full pass on the train set. The latter is therefore considered to be more efficient. One may distinguish between **batch-all**, where all valid triplets from a batch are taken, and **batch-hard**, where for each anchor the hardest positive and negative sample are chosen.
