---
layout: post
title: "How to post notebooks?"
mathjax: true
tags: ["Blogging", "Jekyll", "Jupyter notebooks"]
---

It turns out that blog-posting Jupyter notebooks with [Jekyll](https://jekyllrb.com/) (and the [Beautiful Jekyll](https://beautifuljekyll.com/) theme) is enjoyably simple. One just needs to follow the instructions below.


### 1. Convert notebook

The first step is to convert a Jupyter notebook `notebook.ipynb` to Markdown:
```shell
jupyter nbconvert --to markdown notebook.ipynb
```

This creates a file `notebook.md` which should be renamed according to `yyyy-mm-dd-title.md` and then moved into the `_posts` directory of the website sources.

The notebook's plots are automatically stored in a dedicated folder. They should be transferred to a suitable location, say `assets/images`, and loaded from there.


### 2. Add front matter

In the next step, one needs to add a YAML front matter block at the beginning of the Markdown file. For example, the front matter of this post looks like:
```yaml
---
layout: post
title: "How to post notebooks?"
mathjax: true
tags: ["Blogging", "Jekyll", "Jupyter notebooks"]
---
```

While the entries `layout`, `title` and `tags` are pretty self-explanatory, the third field specifies a custom variable that is further explained in the following.


### 3. Adapt equations

In case one wants to render LaTeX equations with MathJax, one can simply set `mathjax: true` in the front matter. It is remarked that this here involves some changes to standard LaTeX though.

- Instead of using `$` as delimiters for inline equations, double dollar signs `$$` are required. For example, `$$a = 1$$` is rendered as $$a = 1$$.

- For display math mode one can also use the `$$` delimiters, as usual, but they should be enclosed by additional blank lines. The following are the source and render of an example display equation:

    ```latex

    $$
    b = 2
    $$

    ```

    $$
    b = 2
    $$

- Beyond the adaptions above, some other issues might arise. A problem I encountered is that `|`, as in $$p(x \vert y)$$ for instance, has to be replaced with `\vert`. Otherwise it is falsely interpreted as a Markdown table :upside_down_face:
