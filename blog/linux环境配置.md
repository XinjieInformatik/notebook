# 环境配置

## LINUX

### 配置终端颜色/vim
```bash
# vi ~/.bashrc, ~/.zshrc 配置终端颜色
PS1='\[\e[1;32m\]\u@\h\[\e[m\]:\[\e[1;34m\]\W\[\e[1;33m\]\$\[\e[m\] '
```
```bash
# vi ~/.vimrc 配置vim编辑器
set number                                         
colorscheme desert # useless in mac
au BufNewFile,BufRead *.py,*.pyw setf python
set autoindent " same level indent
set smartindent " next level indent
set expandtab
set tabstop=4
set shiftwidth=4
set textwidth=200
set softtabstop=4

set cuc
set cul

set mouse=r
syntax on
set showmatch
set hlsearch
set ignorecase
set wildmenu
set wildmode=longest:list,full
set autoindent
set history=1000
```

```bash
# ~/.bashrc
# cuda setting
export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-11.1/bin:$PATH
export CUDA_HOME=/usr/local/cuda-11.1
# gcc
export LD_LIBRARY_PATH=/usr/local/gcc-5.4.0/lib64:$LD_LIBRARY_PATH
# python path
export PYTHONPATH=/home/users/xinjie.wang/path_to_dir/python:$PYTHONPATH
```