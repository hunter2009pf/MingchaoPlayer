# MingchaoPlayer
Play Mingchao with actions and voice.
## Author's saying
It is a good idea. More importantly, it is great to make the idea work.

## Pose2action
| Key | Pose |
| :----:| :----: | 
| A(left) | 单元格 |
| W(forward) | 单元格 |
| S(back) | 单元格 |
| D(right) | 单元格 |
| T | 单元格 |
| Space(jump) | 单元格 |
| Mouse Left One Click(attack) | 单元格 |
| Mouse Right One Click(escape) | 单元格 |
| Mouse Move Left(turn on left) | 单元格 |
| Mouse Move Right(turn on right) | 单元格 |

1. 走路向前
2. 小幅度左转向走路
3. 大幅度左转向走路
4. 加速跑：手臂斜45°向前（按住鼠标右键3秒）
5. 小幅度右转走路
6. 大幅度右转走路
7. 面向前，向左平移
8. 面向前，向右平移
9. 后退：身体板正后倾斜45°
10. 闪避：身体左后倾 或 身体右后倾
11. 普通攻击：打拳
12. E技能：两手掌合并为中空三角形
13. R大招：基纽特战队队长登场姿势
14. 空格：跳跃

## Technical Planning
1. Use yolov8 to detect and classify human poses.
   ```
   # command line to 
   yolo pose predict model=D:/digital_human/MingchaoPlayer/models/yolo_models/yolov8n-pose.pt source='https://ultralytics.com/images/bus.jpg'
   ```
   动作区别越大，识别准确率越高；闪避可以用双臂交叉挡住脸。
   识别不准的动作有：跳跃、向左平移、向右平移

2. Use Damo, OP or PyAutoGUI to operate mouse and keyboard.
3. Use Whisper to convert the player's real-time audio into text.
4. Text splitting and phrases mapping to commands. 

## Implementation Detail
1. yolov8分类器
   ffmpeg进行视频切帧，结果是一帧帧的图片
   ```
   ffmpeg -i VID20240528093613.mp4 -f image2 ./source/output_%03d.png
   ```

## References
1. https://github.com/CMU-Perceptual-Computing-Lab/openpose
2. https://www.cnblogs.com/yuyingblogs/p/16177798.html
3. https://zh.d2l.ai/chapter_convolutional-neural-networks/index.html
4. https://learn.microsoft.com/zh-cn/windows/ai/windows-ml/tutorials/pytorch-train-model
5. https://github.com/mmakos/HPC
6. https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python?hl=zh-cn
7. ASR System(Whisper): https://blog.csdn.net/lsb2002/article/details/131056566
