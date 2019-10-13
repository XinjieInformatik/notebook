# 立体视觉 Stereo Vision
最近在研究一些双目3D，深度估计的方向，但是中文这方面系统性的知识很少，于是在学习的同时做个总结。这份总结会从传统stereo vision的知识点与数学基础开始，过渡到深度学习的方法。classical -> deeplearning
reference:
http://www.vision.deis.unibo.it/smatt/Seminars/StereoVision.pdf

## 传统 Stereo Vision
<img src="assets/立体视觉_Stereo_Vision-39604de6.png" width="30%" />
<img src="assets/立体视觉_Stereo_Vision-4ebd1630.png" width="60%" />

通过使用使用两个（或多个）相机，如果能在两个图像中找到对应点，则可以通过三角测量(triangulation)来推断深度

<img src="assets/立体视觉_Stereo_Vision-73e255ab.png" width="30%" />
<img src="assets/立体视觉_Stereo_Vision-0118d06f.png" width="60%" />

对极约束(epipolar constraint)指出: 位于（红色）视线的点的对应点位于目标图像像平面πT上的绿线上。

<img src="assets/立体视觉_Stereo_Vision-2ad5aaf6.png" width="70%" />

<img src="assets/立体视觉_Stereo_Vision-a3504486.png" width="70%" />

可以找到平行的相平面，使得πR与πT上的成像点位于同一扫描线上

<img src="assets/立体视觉_Stereo_Vision-d8d48a75.png" width="60%" />

视差d与深度Z的转换：相似三角形
$$ \frac{b}{Z} = \frac{(b - x_R) + x_T}{Z - f} $$
$$ Z = \frac{b * f}{x_R - x_T} = \frac{b * f}{d} $$

<img src="assets/立体视觉_Stereo_Vision-52a04195.png" width="100%" />

离摄像头越近，时差越大

<img src="assets/立体视觉_Stereo_Vision-95341eef.png" width="70%" />

每个视差值对应一个深度平面, 深度Z的范围Horopter受限于视差d

立体视觉算法总体分为如下4个步骤
1) Matching cost computation 匹配损失计算
2) Cost aggregation 损失聚合
3) Disparity computation/optimization 视差计算/优化
4) Disparity refinement 视差细化

在1之前还有一些前处理的方法，例如：Laplacian of Gaussian (LoG) filtering, Subtraction of mean values computed in nearby pixels, Bilateral filtering, Census transform

1) Matching cost computation
