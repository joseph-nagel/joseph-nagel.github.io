---
layout: post
title: "Pre-commit hooks"
tags: ["Coding", "Python", "Git hooks", "Pre-commit hooks"]
---

This post gathers a few short notes on Git pre-commit hooks. Before changes are committed to a Git repository, such hooks may automatically check or format the new code. In this way, one can enforce coding standards and best practices, which helps to maintain code quality and consistency throughout the development process.

It has to be noted that pre-commit hooks should be used in addition to CI pipelines, not as a replacement. Pre-commit hooks run locally on a developer's machine and provide immediate feedback during a commit. They need to be fast and therefore limited to a small set of stylistic and syntactic checks. CI runs remotely after code is pushed or a PR is created. This should conduct a more thorough and authoritative validation, including for instance unit tests, before changes are finally merged.

Generally speaking, Git hooks are just executable scripts that are triggered by certain events. These scripts are located in the `.git/hooks/` directory of a Git repository. Usually, there should be already a couple of built-in examples at this location, e.g. `pre-commit.sample` or `pre-push.sample`. The hooks can be enabled by simply removing the `.sample` suffix from the file name.

There are different ways to set up and manage Git hooks. While it is entirely possible to write custom scripts for each hook, it is often more convenient to rely on a dedicated framework for configuring hooks. The most popular choice is [pre-commit](https://pre-commit.com/). A modern and fully compatible alternative is established by [prek](https://prek.j178.dev/). The remainder of this post focuses on the common `pre-commit` framework.


## Instructions

1. Install `pre-commit`. If it is not yet installed through the standard project dependencies, one may run for example:
    ```bash
    pip install pre-commit
    ```
2. Create a config file `.pre-commit-config.yaml`. A sample configuration can be generated with:
    ```bash
    pre-commit sample-config
    ```
3. Set up a Git hook script `.git/hooks/pre-commit` that executes before each commit:
    ```bash
    pre-commit install
    ```
4. Optionally, since this runs the hooks on the staged files only, one may want to run them once on all files:
    ```bash
    pre-commit run --all-files
    ```

After this setup, each time a `git commit` is made, the pre-commit hooks will be executed automatically. If any check fails, the commit will be aborted, and the user will be prompted to fix the issues before trying again. In case one wants to skip the Git pre-commit hooks for a single commit, one can use the `--no-verify` flag, for instance `git commit --no-verify -m "Commit without verification"`.


## Example config

We conclude this post with an example configuration for the [Ruff pre-commit hook](https://github.com/astral-sh/ruff-pre-commit). A simple configuration file `.pre-commit-config.yaml` for linting and formatting at each commit could look like this:
```yaml
repos:
- repo: https://github.com/astral-sh/ruff-pre-commit
  rev: v0.16.6
  hooks:
    - id: ruff-check
      args: [ --fix ]
    - id: ruff-format
```
