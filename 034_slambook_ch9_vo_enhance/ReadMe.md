# 034 slambook ch9 vo enhance

对"视觉SLAM十四讲"中的里程计实验的一个小增强.

以 TUM 格式输出成轨迹文件,用于 evo 中分析,像 ORB-SLAM2那样.

评估 APE 使用指令

```
evo_ape tum ./groundtruth.txt ./CameraPoses.txt -a
```

评估RPE平移部分使用指令

```
evo_rpe tum ./groundtruth.txt ./CameraPoses.txt -a
```

评估RPE旋转部分使用指令

```
evo_rpe tum ./groundtruth.txt ./CameraPoses.txt --pose_relation angle_deg --delta 1 --delta_unit m -a
```

