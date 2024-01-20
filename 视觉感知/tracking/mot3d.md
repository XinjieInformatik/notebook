# MOT3d Tracker

## Simple Track

![simple track pipeline](2023-05-13-14-08-16.png)

1. 检测结果预处理 (score filter / NMS)
2. 预测轨迹在K-1时刻在K帧的位置 (卡尔曼滤波)
3. 关联`K-1轨迹在K帧的预测` 与 `K帧检测结果` 的关联 (IoU / L2距离)
4. 生命周期管理，维护轨迹的 `birth`, `alive`, `no_asso`, `dead` 状态

