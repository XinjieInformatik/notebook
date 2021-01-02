# Jupyter-lab

最近在做一些数据分析的工作，发现思路需要走走停停，用Jupyter-lab比其他IDE更方便，远程连接服务器的时候Jupyter-lab也很方便，所以在这里整理一下常用插件。

首先安装 Jupyter-lab
```sh
pip install jupyterlab
```
```sh
conda install -c conda-forge jupyterlab
```

然后安装nodejs，为了后续安装与管理插件。使用 conda-forge 获得更高版本nodejs
```sh
conda install -c conda-forge nodejs
```

然后打开Jupyter - Settings - Advanced Settings Editor，在 User Preferences 中 打 "enabled": true，保存退出，然后就可以在左侧搜索插件安装了。

推荐插件：
- jupyterlab-manager
- @jupyterlab/toc
- @krassowski/jupyterlab_go_to_definition
- jupyter-matplotlib
- jupyterlab-drawio
- jupyterlab_autoversion
- jupyterlab-execute-time
- jupyterlab-system-monitor
- collapsible_headings

todo:
- jupyterlab_code_formatter
- @jupyterlab/debugger
conda install xeus-python -c conda-forge
[目前还是有bug](https://github.com/jupyterlab/debugger)


Jupyter lab 的启动开始文件路径就是终端打开它的位置。

常用快捷键
- 选中cell时，Ctrl+Enter 运行该cell
- Esc进入命令模式，Shift+Enter，运行该cell并下一行
- Esc进入命令模式，a，下方插入cell
- Esc进入命令模式，b，上方插入cell
- Esc进入命令模式，dd，删除cell
- Esc进入命令模式，c，复制cell
- Esc进入命令模式，v，粘贴cell
- Esc进入命令模式，m，cell设为markdown
- Esc进入命令模式，y，cell设为code


qgrid — DataFrame 交互
https://github.com/quantopian/qgrid
```python
import qgrid  
qgrid_widget = qgrid.show_grid (df, show_toolbar=True)
qgrid_widget
```

[切换kernel](https://stackoverflow.com/questions/37891550/jupyter-notebook-running-kernel-in-different-env)
```sh
conda activate ENVNAME
pip install ipykernel
python -m ipykernel install --user --name ENVNAME --display-name "displayname"
```

[清华源](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)
安装`xeus-python`时，清华源里没有这个资源，把`.condarc`备份一下删了，`conda clean -i` 清除索引缓存，装好了后可以再换回来

remove jupyterlab
```sh
conda uninstall notebook nbconvert nbformat ipykernel ipywidgets qtconsole traitlets tornado jupyter_* ipython_genutils jinja2 -y
conda clean --all
```

reference:
- [数据挖掘：工具篇（一）—— Jupyter Lab 配置环境](https://liketea.xyz/%E6%95%B0%E6%8D%AE%E6%8C%96%E6%8E%98/%E6%95%B0%E6%8D%AE%E6%8C%96%E6%8E%98/%E6%95%B0%E6%8D%AE%E6%8C%96%E6%8E%98%EF%BC%9A%E5%B7%A5%E5%85%B7%E7%AF%87%EF%BC%88%E4%B8%80%EF%BC%89%E2%80%94%E2%80%94%20Jupyter%20Lab%20%E9%85%8D%E7%BD%AE%E7%8E%AF%E5%A2%83/)
