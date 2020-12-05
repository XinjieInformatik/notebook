---
title: 搭建个人博客 github + hexo + NexT
date: 2020-09-09 17:37:47
categories:
- hexo博客搭建
tags:
- hexo博客搭建
top: false

---

# 搭建个人博客 hexo

这篇文章介绍从0到1搭建个人博客, 基于 Github.io + hexo + NexT，完全免费也美观，对于个人博客来说完全够了，半天内可以搭建并自定义完成个人博客，推荐！
<!--more-->

- 基于github与hexo搭建博客，完全免费
- hexo提供许多现成主题模板
- 经过配置后可使用latex。

## 建立 hexo 博客站点步骤

- github 新建一个repository， 名称为 `你的GitHub用户名.github.io`
- 确保NodeJS的安装目录与Git的安装目录在同一个文件夹下（[参考](https://stackoverflow.com/questions/45513441/npm-command-not-found-in-windows-10)）
- 下载并安装NodeJS
- 下载并安装Git for Windows
- 打开Git bash 配置Git的用户名与邮箱为github用户名与邮箱，生成本地ssh并复制到github ssh上 （第一次使用Git设置，如果已经设置过，可以跳过该步骤）
- 选定一个文件夹作为存放博客文件的目录，以管理员身份打开Git bash， 并 cd 到该目录
- 导入nodejs路径到`git export PATH=$PATH:"/D/path_to_nodejs"` (windows path)
- 输入命令 `npm install hexo-cli -g`
- 输入命令 `hexo init blog`
- 输入命令 `cd blog`
- 输入命令 `npm install`
- 输入命令 `hexo s`
    s是server的缩写，表示启动本地服务器。默认情况下，访问网址是：http://localhost:4000/ 。如果成功的话可以看见官方的默认页面。
- 在blog文件夹下编辑_config.yml, 找到相应字段，修改如下

```yaml
url: https://你的GitHub用户名.github.io/
deploy:
  type: git
  repo: https://github.com/你的GitHub用户名/你的GitHub用户名.github.io.git
  branch: master
```

- 输入命令

```sh
npm install hexo-deployer-git --save
hexo clean
hexo g -d
```

- 10几20分钟后打开浏览器，输入你的网站地址：https://你的GitHub用户名.github.io/，就可以看到你的网站已经上线了

## 新建并发布文章

```sh
hexo n "newblog"
hexo clean
hexo g -d
```

在Git bash 命令行输入 hexo n "newblog", 就可以看到在blog/source/_posts目录下多了一个newblog.md文件。写完之后生成、部署即可(运行 hexo g -d)。
要删除文章的话，直接把源文件删除即可（确保至少有一篇文章存在，否则可能出错）
同步到github.io通常要等待一段时间

参考：[一小时搭建完自己的个人网站](https://zhuanlan.zhihu.com/p/78467553)

## 常用的配置

### 更换主题

Hexo有许多好看的主题，以NexT主题为例（使用的是NexT 7.8.0版本）：
下载主题：Releases · theme-next/hexo-theme-next
解压所下载的压缩包至站点的 themes 目录下，并将解压后的文件夹名称（hexo-theme-next-x.x.x）更改为 next
打开站点配置文件_config.yml，找到 theme 字段，并将其值更改为 next ；找到 language 字段，并将其值更改为 zh-CN（在themes/next/languages目录下可查看主题支持的语言）
在切换主题之后，使用 hexo clean 命令来清除 Hexo 的缓存

### LaTeX支持

[配置latex](https://theme-next.js.org/docs/third-party-services/math-equations.html)

### 插入图片

安装一个能上传本地图片的插件：

```sh
npm install https://github.com/CodeFalling/hexo-asset-image --save
```

我一般会atom写markdown，多个blog的图片统一放在assets中

### 文章添加tags&category

设置blog\scaffolds\post, 每次 hexo n "new_blog" 默认会出现的标题

```sh
---
title: 两数之和
date: 2020-09-09 12:12:57
categories:
- leetcode
tags:
- 双指针
- 哈希表
---
```

### 更换博客图片背景，设置透明度

在 根目录/source下新建_data/styles.styl, 设置如下,
背景图片放在 根目录/source/images/
字体颜色等均可在这里改，覆盖全局参数

```sh
// Custom styles.
// 整体背景设置
body {
  background:url(/images/1_compress.jpg);// 设定背景图片,images同处于blog/source文件夹下
 	background-repeat: no-repeat;// 设定背景图片非重复填充
    background-attachment:fixed;// 设置背景图片不随页面滚动
    background-position:50% 50%;// 设置背景图片位置
  background-size: cover// 设置保持图像的纵横比并将图像缩放成将完全覆盖背景定位区域的最小大小
}

// 文章内容的透明度设置
.content-wrap {
  opacity: 0.86;
  color: #000000;
}

// sidebar侧边工具栏样式属性
.sidebar{
  opacity: 0.86
  color: #000000;
}

// 页面头样式属性
.header-inner {
  background: rgba(255,255,255,0.86);
  color: #000000;
}

// 页脚颜色
.footer {
  color: #DCDCDC;
}

// 搜索框（local-search）的透明度设置
.popup {
  opacity: 0.86;
}


// 向上箭头
.back-to-top {
    opacity: 1;
    line-height: 2.8;
    right: 35px;
    padding-right: 5px;
    padding-left: 5px;
    padding-top: 2.5px;
    padding-bottom: 2.5px;
    background-color: rgba(28, 28, 28, 1);
    border-radius: 5px;
}
```

### next主题美化与第三方插件

参考：

- [NexT文档](https://theme-next.js.org/docs/)
- [这个博客里有一系列教程,推荐!注意有些设置是低版本的](https://tding.top/archives/42c38b10.html)
- [hexo NexT 主题美化，注意有些设置是低版本的](http://eternalzttz.com/hexo-next.html)
- [Hexo-NexT 添加打字特效、鼠标点击特效](https://tding.top/archives/58cff12b.html)
- [时钟特效](https://jrbcode.gitee.io/posts/c13e56cd.html)
- [点击文章后从头开始阅读全文](https://blog.csdn.net/weizhixiang/article/details/105112467)
- [next主题添加背景图片(无custom.styl情况)](https://blog.csdn.net/chrishly3/article/details/103992492)


### 性能问题

一开始我启用了 [tidio](https://www.tidio.com/panel/settings/live-chat/appearance) 在线聊天，busuanzi 站点访问统计
分析网页加载性能时，瓶颈主要在他们。介于对我来说，他们没那么重要，我就把他们都disable了，网页加载快了很多。

### 关于TOC中文跳转失效的bug

NexT 7.3 以后版本自带toc了，在主题的config.yml中enable即可。

```sh
toc:
  enable: true
```

但是遇到的问题是，当文章目录 toc 含有中文时，点击左侧目录栏无法正常跳转，查了很多资料，都没能解决（例如卸载hexo-toc（这东西就不用装）,标题文件命名规则等）。结果发现原来是js解析id的问题。。
打开node_modules\hexo\lib\plugins\helper\toc.js，修改如下

```js
    // const href = id ? `#${encodeURL(id)}` : null;
    const href = id ? `#${id}` : null;
```

注意如果标题中含有空格，会被解析成-。但这些都不影响使用，把encodeURL去掉，就一切正常了。

### 禁止部分markdown gitalk 评论

题头添加 comments: false

```
---
title: about
date: 2020-09-09 13:26:08
comments: false
---
```
