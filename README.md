# Personal website

This is the repository of my personal website made with [GitHub Pages](https://docs.github.com/en/pages),
[Jekyll](https://jekyllrb.com/) and [Beautiful Jekyll](https://beautifuljekyll.com/).


## Instructions

- Install requirements (on Fedora Linux):
    ```bash
    sudo dnf install ruby ruby-devel openssl-devel redhat-rpm-config gcc-c++ @development-tools
    ```

- Install Jekyll:
    ```
    gem install jekyll bundler
    ```

- Create new site in existing directory:
    ```bash
    jekyll new . --force
    ```

- Preview locally:
    ```bash
    bundle exec jekyll serve --livereload
    ```

- Update gems from `Gemfile`:
    ```bash
    bundle update
    ```

<!-- - Create `favicon.ico` from `Photo.jpg`:
    ```bash
    magick assets/images/Photo.jpg -define icon:auto-resize=16,32,48,64 -flatten -colors 256 favicon.ico
    ``` -->
