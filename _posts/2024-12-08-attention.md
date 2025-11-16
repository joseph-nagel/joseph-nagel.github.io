---
layout: post
title: Attention mechanism
mathjax: true
tags: [Introduction, Attention mechanism, Transformer, Vision transformer]
thumbnail-img: https://raw.githubusercontent.com/joseph-nagel/attention-mechanism/main/assets/attention.svg
gh-repo: joseph-nagel/attention-mechanism
gh-badge: [star, fork, follow]
---

This blog post gives a brief introduction to **attention** in deep neural nets. Attention establishes a mechanism that allows a model to make predictions based on selectively attending to different items of an input sequence. It can be employed as a pretty generic modeling layer for problems with a sequential structure (and beyond).

Attention has been developed in language processing [[Bahdanau et al., 2015](https://arxiv.org/abs/1409.0473); [Luong et al., 2015](https://arxiv.org/abs/1508.04025)]. The **transformer architecture** from the *Attention Is All You Need*-paper [[Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)] is seen as a breakthrough in this field. It overcomes some shortcomings of recurrent neural networks. The **vision transformer** [[Dosovitskiy et al., 2021](https://arxiv.org/abs/2010.11929)] applies self-attention to tokenized image patches. Current reviews of different attention variants are provided in [[Brauwers and Frasincar, 2022](https://arxiv.org/abs/2203.14263); [Lin et al., 2022](https://doi.org/10.1016/j.aiopen.2022.10.001)].


## Dot-product attention

We start with a very short discussion of the equation and interpretation of the **scaled dot-product attention** in [[Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)]. Adopting the convention that the main objects are row vectors organized in matrices, this influential attention variant is usually written as

$$
\operatorname{Attention}(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{\boldsymbol{V}}) =
\operatorname{Softmax} \left( \frac{\boldsymbol{Q} \boldsymbol{K}^\top}{\sqrt{d_k}} \right) \boldsymbol{V}.
$$

Here, **queries**, **keys** and **values** are the rows of $$Q \in \mathbb{R}^{m \times d_k}$$, $$K \in \mathbb{R}^{n \times d_k}$$ and $$V \in \mathbb{R}^{n \times d_v}$$, respectively. One easily sees that $$\boldsymbol{Q} \boldsymbol{K}^\top \in \mathbb{R}^{m \times n}$$ is a matrix of query-key dot-products

$$
\boldsymbol{Q} \boldsymbol{K}^\top =
\begin{pmatrix}
\text{-} & \boldsymbol{q}_1 & \text{-} \\
\text{-} & \boldsymbol{q}_2 & \text{-} \\
\vdots & \vdots & \vdots \\
\text{-} & \boldsymbol{q}_m & \text{-} \\
\end{pmatrix}
\begin{pmatrix}
\vert & \vert & \ldots & \vert \\
\boldsymbol{k}_1 & \boldsymbol{k}_2 & \ldots & \boldsymbol{k}_n \\
\vert & \vert & \ldots & \vert \\
\end{pmatrix} =
\begin{pmatrix}
\boldsymbol{q}_1 \cdot \boldsymbol{k}_1 & \boldsymbol{q}_1 \cdot \boldsymbol{k}_2 &
\ldots & \boldsymbol{q}_1 \cdot \boldsymbol{k}_n \\
\boldsymbol{q}_2 \cdot \boldsymbol{k}_1 & \boldsymbol{q}_2 \cdot \boldsymbol{k}_2 &
\ldots & \boldsymbol{q}_2 \cdot \boldsymbol{k}_n \\
\vdots & \vdots & \ddots & \vdots \\
\boldsymbol{q}_m \cdot \boldsymbol{k}_1 & \boldsymbol{q}_m \cdot \boldsymbol{k}_2 &
\ldots & \boldsymbol{q}_m \cdot \boldsymbol{k}_n \\
\end{pmatrix}.
$$

It establishes a measure of vector similarity between the queries and keys. Each element of this matrix is divided by $$\sqrt{d_k}$$. This scaling avoids extremely small gradients for large $$d_k$$ in the following softmax operation. The softmax is applied over each row separately. Finally, $$\boldsymbol{W} = \operatorname{Softmax} \left( d_k^{-1/2} \boldsymbol{Q} \boldsymbol{K}^\top \right)$$ is right-multiplied by the values. This yields

$$
\operatorname{Softmax} \left( \frac{\boldsymbol{Q} \boldsymbol{K}^\top}{\sqrt{d_k}} \right) \boldsymbol{V} =
\begin{pmatrix}
w_{11} & w_{12} & \ldots & w_{1n} \\
w_{21} & w_{22} & \ldots & w_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
w_{m1} & w_{m2} & \ldots & w_{mn} \\
\end{pmatrix}
\begin{pmatrix}
\text{-} & \boldsymbol{v}_1 & \text{-} \\
\text{-} & \boldsymbol{v}_2 & \text{-} \\
\vdots & \vdots & \vdots \\
\text{-} & \boldsymbol{v}_n & \text{-} \\
\end{pmatrix} =
\begin{pmatrix}
\text{-} & \sum_{j=1}^n w_{1j} \boldsymbol{v}_j & \text{-} \\
\text{-} & \sum_{j=1}^n w_{2j} \boldsymbol{v}_j & \text{-} \\
\vdots & \vdots & \vdots \\
\text{-} & \sum_{j=1}^n w_{mj} \boldsymbol{v}_j & \text{-} \\
\end{pmatrix}.
$$

The matrix $$d_k^{-1/2} \boldsymbol{Q} \boldsymbol{K}^\top$$ is sometimes referred to as **alignment scores**, while $$\boldsymbol{W}$$ is called the **attention weights**. In each row $$i \in \{1,\ldots,m\}$$ one has that the weights sum to one $$\sum_{j=1}^n w_{ij} = 1$$. Hence, $$\operatorname{Attention}(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{\boldsymbol{V}}) \in \mathbb{R}^{m \times d_v}$$ can be seen as a relevance-weighted average of the values.

Let us further investigate a single row $$(w_{i1}, \ldots, w_{in})$$ of the attention weight matrix. It describes how the query vector $$\boldsymbol{q}_i$$ is matched against (or attends to) all possible keys $$\boldsymbol{k}_j$$ with $$j=1,\ldots,n$$. This highlights the analogy to information retrieval. The row vector of attention weights is given as

$$
(w_{i1}, \ldots, w_{in}) = \operatorname{Softmax} \left(
\frac{\boldsymbol{q}_i \cdot \boldsymbol{k}_1}{\sqrt{d_k}}, \ldots,
\frac{\boldsymbol{q}_i \cdot \boldsymbol{k}_n}{\sqrt{d_k}} \right) =
\frac{\left( \exp \left( \frac{\boldsymbol{q}_i \cdot \boldsymbol{k}_1}{\sqrt{d_k}} \right), \ldots,
\exp \left( \frac{\boldsymbol{q}_i \cdot \boldsymbol{k}_n}{\sqrt{d_k}} \right) \right)}
{\sum_{j=1}^n \exp \left( \frac{\boldsymbol{q}_i \cdot \boldsymbol{k}_j}{\sqrt{d_k}} \right)}.
$$


## Self-attention

The attention discussed above starts from given queries, keys and values. These objects are often constructed through the multiplication of row vectors with three weight matrices $$\boldsymbol{W}_q \in \mathbb{R}^{d_x \times d_k}$$, $$\boldsymbol{W}_k \in \mathbb{R}^{d_x \times d_k}$$ and $$\boldsymbol{W}_v \in \mathbb{R}^{d_x \times d_v}$$. For example, for a single input $$\boldsymbol{x} \in \mathbb{R}^{1 \times d_x}$$, they would be respectively given as $$\boldsymbol{q} = \boldsymbol{x} \boldsymbol{W}_q$$, $$\boldsymbol{k} = \boldsymbol{x} \boldsymbol{W}_k$$ and $$\boldsymbol{v} = \boldsymbol{x} \boldsymbol{W}_v$$. For multiple inputs $$\boldsymbol{X} \in \mathbb{R}^{m \times d_x}$$ one can similarly write

$$
\boldsymbol{Q} = \boldsymbol{X} \boldsymbol{W}_q, \quad
\boldsymbol{K} = \boldsymbol{X} \boldsymbol{W}_k, \quad
\boldsymbol{V} = \boldsymbol{X} \boldsymbol{W}_v.
$$

When queries, keys, and values are computed for a single sequence, say its $$m$$ elements are stacked in row-wise fashion so as to construct $$\boldsymbol{X}$$, the so-called **self-attention** is given by $$\operatorname{Attention}(\boldsymbol{X} \boldsymbol{W}_q, \boldsymbol{X} \boldsymbol{W}_k, \boldsymbol{X} \boldsymbol{W}_v) \in \mathbb{R}^{m \times d_v}$$. It relates the items at different positions from a single sequence to each other.


## Cross-attention

More generally, one may also connect items from two different sequences though. This is called **cross-attention**. The queries $$\boldsymbol{Q} = \boldsymbol{X} \boldsymbol{W}_q$$ are given as before. The sequence $$\boldsymbol{X} \in \mathbb{R}^{m \times d_x}$$ and weight matrix $$\boldsymbol{W}_q \in \mathbb{R}^{d_x \times d_k}$$ are unchanged. Another sequence $$\boldsymbol{Y} \in \mathbb{R}^{n \times d_y}$$ is used in order to compute the keys $$\boldsymbol{K} = \boldsymbol{Y} \boldsymbol{W}_k$$ and values $$\boldsymbol{V} = \boldsymbol{Y} \boldsymbol{W}_v$$. Here, the weight matrices $$\boldsymbol{W}_k \in \mathbb{R}^{d_y \times d_k}$$ and $$\boldsymbol{W}_v \in \mathbb{R}^{d_y \times d_v}$$ are correspondingly shaped.

<img src="https://raw.githubusercontent.com/joseph-nagel/attention-mechanism/main/assets/attention.svg" alt="The scaled dot-product (cross) attention mechanism is visualized" title="Scaled dot-product (cross) attention" height="250">


## Multi-head attention

A common extension to the mechanism presented above is **multi-head attention**. Here, one simply uses multiple independent attentions with different weight matrices. Their outputs are eventually concatenated and linearly transformed to the desired shape. Letting $$\mathrm{head}_i = \operatorname{Attention}(\boldsymbol{X}_q \boldsymbol{W}_q, \boldsymbol{X}_k \boldsymbol{W}_k, \boldsymbol{X}_v \boldsymbol{W}_v)$$ represent a single attention head (where different input sequences may be used for the queries, keys and values), the multi-head variant is often written as

$$
\operatorname{MultiHeadAttention}(\boldsymbol{X}_q, \boldsymbol{X}_k, \boldsymbol{X}_v) =
\operatorname{Concat}(\mathrm{head}_1, \ldots, \mathrm{head}_h) \boldsymbol{W}_o.
$$


## Positional encoding

It has to be noted that the attention mechanism ignores the order of the input sequence. It can actually be seen as an operation on sets rather than sequences. But one can rely on a so-called **positional encoding** in order to inject and process order information. Such an encoding $$\boldsymbol{P} \in \mathbb{R}^{m \times d_x}$$ is often constructed such that it can be added to the token embeddings $$\boldsymbol{X} \in \mathbb{R}^{m \times d_x}$$. Then, $$\boldsymbol{X} + \boldsymbol{P}$$ can be plugged in instead of $$\boldsymbol{X}$$ in all downstream operations that follow.

A **sinusoidal encoding** for sequence positions has already been proposed in [[Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)]. Its usage is now commonplace. Given that the number of dimensions $$d_x$$ is even, the $$t$$-th row of this encoding $$\boldsymbol{p}_t \in \mathbb{R}^{1 \times d_x}$$ is given as

$$
\boldsymbol{p}_t^\top =
\begin{pmatrix}
\sin(\omega_1 \cdot t) \\ \cos(\omega_1 \cdot t) \\[0.4em]
\sin(\omega_2 \cdot t) \\ \cos(\omega_2 \cdot t) \\
\vdots \\
\sin(\omega_{d_x/2} \cdot t) \\ \cos(\omega_{d_x/2} \cdot t)
\end{pmatrix}, \quad
\omega_k = \frac{1}{10000^{2k/d_x}}.
$$

This embedding is fixed by construction and is not learned from data. Of course, other embeddings may be used as well. A more general overview of position information in transformers is provided in [[Dufter et al., 2022](https://doi.org/10.1162/coli_a_00445)].

<img src="https://raw.githubusercontent.com/joseph-nagel/attention-mechanism/main/assets/sinusoidal.svg" alt="Sinusoidal encoding of spatial positions or times" title="Sinusoidal encoding" height="250">


## Transformer

The basic building blocks of the transformer model are multi-head attention and feed-forward neural network layers. Neither recurrence nor convolutions are used. The layers are stacked and organized into an **encoder-decoder** architecture. The sinusoidal encoding from above is added to the inputs of the encoder and decoder in order to integrate positioning info. Residual connections and layer normalization are employed within each part. Learning rate warm-up is typically applied during the first few training epochs.


## Masking

It is widespread practice to mask out certain elements of the attention matrix. This is often called **masked attention**. A common example is **causal masking** where tokens are not allowed to attend to the "future". This can be easily realized with a lower triangular mask.

Beyond that, another motivation of masking is to reduce the computational complexity. The global all-to-all attention suffers from a $$\mathcal{O}(n^2)$$ time and memory complexity ($$n$$ is the sequence length). In order to enable long context lengths, one may restrict the attention to local neighborhoods of each token through a **sliding windows** scheme [[Beltagy et al., 2020](https://arxiv.org/abs/2004.05150)]. This reduces the complexity to $$\mathcal{O}(n \cdot w)$$ (the window size is denoted as $$w$$).

The scaling problem also emerges for the ViT. Its complexity is quadratic in the image size. A **shifted windows** partitioning scheme for the ViT has been presented in [[Liu et al., 2021](https://arxiv.org/abs/2103.14030)]. It limits self-attention to non-overlapping windows, while it still allows for certain cross-window connections. This approach features a complexity that is only linear in the image size.
