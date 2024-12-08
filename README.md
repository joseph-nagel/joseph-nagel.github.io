# Personal website

This is the repository of my personal website made with [GitHub Pages](https://docs.github.com/en/pages),
[Jekyll](https://jekyllrb.com/) and [Beautiful Jekyll](https://beautifuljekyll.com/).


## Instructions

- Install requirements (on Fedora Linux):
    ```
    sudo dnf install ruby ruby-devel openssl-devel redhat-rpm-config gcc-c++ @development-tools
    ```

- Install Jekyll:
    ```
    gem install jekyll bundler
    ```

- Create new site in existing directory:
    ```
    jekyll new . --force
    ```

- Preview locally:
    ```
    bundle exec jekyll serve --livereload
    ```

- Update gems from `Gemfile`:
    ```
    bundle update
    ```

- Convert Jupyter notebook to Markdown:
    ```
    jupyter nbconvert --to markdown intro.ipynb
    ```

