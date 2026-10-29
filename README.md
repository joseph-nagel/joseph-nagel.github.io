# Personal website

This is the repository of my personal website and blog made with [GitHub Pages](https://docs.github.com/en/pages), [Jekyll](https://jekyllrb.com/) and the [Chirpy theme](https://chirpy.cotes.page).


## Instructions

- Follow the instructions from [Chirpy's getting-started guide](https://chirpy.cotes.page/posts/getting-started/)
- Edit [`_config.yml`](_config.yml) and, if needed, [`contact.yml`](_data/contact.yml) and [`share.yml`](_data/share.yml)
- Further customize the [Chirpy theme](https://github.com/cotes2020/jekyll-theme-chirpy):
    - Copy [`_layouts/post.html`](_layouts/post.html) from the theme and comment out the preview image at the top of the post
    - Copy [`assets/css/jekyll-theme-chirpy.scss`](assets/css/jekyll-theme-chirpy.scss) add modifications for square preview images


## Other helpful stuff

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

<!-- - Use a specific Ruby version:
    ```
    sudo dnf install rbenv
    rbenv install 3.1.7
    rbenv global 3.1.7
    eval "$(rbenv init -)"
    gem install bundler -v 2.3.27
    ``` -->
