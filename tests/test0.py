import os
from ultralytics import YOLO

labels = [
    "攻击",
    "后退",
    "闪避",
    "向左平移",
    "向右平移",
    "前进",
    "跳跃",
    "E技能",
    "跑步",
    "快速左转",
    "缓慢左转",
    "快速右转",
    "缓慢右转",
    "R技能",
]

if __name__=="__main__":
    model = YOLO("D:/digital_human/yolo_test_20240528/trained_models/best.pt")
    
    directory = 'D:/digital_human/yolo_test_20240528/images_for_eval'

    # 使用os.walk()遍历目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 构造完整的文件路径
            filepath = os.path.join(root, file)
            print(filepath)
            # 这里可以添加处理文件的代码
            results = model(filepath, conf=0.7)
            # print(results[0].probs.data)
            # Now convert to NumPy array
            numpy_array = results[0].probs.data.numpy()
            max_prob = numpy_array[0]
            max_idx = 0
            for i in range(1, 14):
                if numpy_array[i] > max_prob:
                    max_prob = numpy_array[i]
                    max_idx = i
            print(f"文件{file}的类型是：{labels[max_idx]}，置信度是{max_prob}")
    
    # results = model("D:/digital_human/yolo_test_20240528/jack_dataset/test/attack/output_3654.png", conf=0.7)
    # print(results[0].probs.data)

# 第一轮测试结果：
# image 1/1 D:\digital_human\yolo_test_20240528\images_for_eval\attack0.png: 1024x1024 attack 0.99, back 0.01, escape 0.00, jump 0.00, ultraSkill 0.00, 992.3ms
# Speed: 55.7ms preprocess, 992.3ms inference, 0.0ms postprocess per image at shape (1, 3, 1024, 1024)
# 文件attack0.png的类型是：攻击，置信度是0.9941654205322266
# D:/digital_human/yolo_test_20240528/images_for_eval\back0.png

# image 1/1 D:\digital_human\yolo_test_20240528\images_for_eval\back0.png: 1024x1024 back 1.00, escape 0.00, ultraSkill 0.00, turnOnLeftQuickly 0.00, faceForwardMoveLeft 0.00, 964.1ms
# Speed: 55.5ms preprocess, 964.1ms inference, 0.0ms postprocess per image at shape (1, 3, 1024, 1024)
# 文件back0.png的类型是：后退，置信度是0.9999996423721313
# D:/digital_human/yolo_test_20240528/images_for_eval\escape0.png

# image 1/1 D:\digital_human\yolo_test_20240528\images_for_eval\escape0.png: 1024x1024 escape 1.00, back 0.00, ultraSkill 0.00, turnOnRightQuickly 0.00, faceForwardMoveLeft 0.00, 1176.1ms
# Speed: 54.5ms preprocess, 1176.1ms inference, 0.0ms postprocess per image at shape (1, 3, 1024, 1024)
# 文件escape0.png的类型是：闪避，置信度是0.999484658241272
# D:/digital_human/yolo_test_20240528/images_for_eval\forward0.png

# image 1/1 D:\digital_human\yolo_test_20240528\images_for_eval\forward0.png: 1024x1024 forward 0.51, jump 0.17, attack 0.14, turnOnLeftSlowly 0.10, escape 0.04, 1353.3ms
# Speed: 65.6ms preprocess, 1353.3ms inference, 0.0ms postprocess per image at shape (1, 3, 1024, 1024)
# 文件forward0.png的类型是：前进，置信度是0.5144027471542358
# D:/digital_human/yolo_test_20240528/images_for_eval\jump0.png

# image 1/1 D:\digital_human\yolo_test_20240528\images_for_eval\jump0.png: 1024x1024 escape 0.89, back 0.08, attack 0.01, jump 0.01, speedUp 0.00, 1259.5ms
# Speed: 77.4ms preprocess, 1259.5ms inference, 0.0ms postprocess per image at shape (1, 3, 1024, 1024)
# 文件jump0.png的类型是：闪避，置信度是0.8857537508010864
# D:/digital_human/yolo_test_20240528/images_for_eval\move_left0.png

# image 1/1 D:\digital_human\yolo_test_20240528\images_for_eval\move_left0.png: 1024x1024 escape 0.89, faceForwardMoveLeft 0.11, back 0.00, faceForwardMoveRight 0.00, ultraSkill 0.00, 1261.8ms
# Speed: 70.8ms preprocess, 1261.8ms inference, 0.0ms postprocess per image at shape (1, 3, 1024, 1024)
# 文件move_left0.png的类型是：闪避，置信度是0.8866881132125854
# D:/digital_human/yolo_test_20240528/images_for_eval\move_right0.png

# image 1/1 D:\digital_human\yolo_test_20240528\images_for_eval\move_right0.png: 1024x1024 faceForwardMoveLeft 0.43, faceForwardMoveRight 0.36, turnOnLeftQuickly 0.08, back 0.07, escape 0.04, 1292.1ms
# Speed: 69.9ms preprocess, 1292.1ms inference, 0.0ms postprocess per image at shape (1, 3, 1024, 1024)
# 文件move_right0.png的类型是：向左平移，置信度是0.4285653233528137
# D:/digital_human/yolo_test_20240528/images_for_eval\quick_left0.png

# image 1/1 D:\digital_human\yolo_test_20240528\images_for_eval\quick_left0.png: 1024x1024 turnOnLeftQuickly 0.97, turnOnRightQuickly 0.03, escape 0.00, back 0.00, ultraSkill 0.00, 1237.6ms
# Speed: 68.3ms preprocess, 1237.6ms inference, 0.0ms postprocess per image at shape (1, 3, 1024, 1024)
# 文件quick_left0.png的类型是：快速左转，置信度是0.9695559740066528
# D:/digital_human/yolo_test_20240528/images_for_eval\quick_right0.png

# image 1/1 D:\digital_human\yolo_test_20240528\images_for_eval\quick_right0.png: 1024x1024 turnOnRightQuickly 0.98, escape 0.01, turnOnRightSlowly 0.00, faceForwardMoveRight 0.00, faceForwardMoveLeft 0.00, 1299.4ms
# Speed: 72.4ms preprocess, 1299.4ms inference, 0.0ms postprocess per image at shape (1, 3, 1024, 1024)
# 文件quick_right0.png的类型是：快速右转，置信度是0.9828925132751465
# D:/digital_human/yolo_test_20240528/images_for_eval\slow_left0.png

# image 1/1 D:\digital_human\yolo_test_20240528\images_for_eval\slow_left0.png: 1024x1024 turnOnLeftSlowly 0.93, turnOnRightSlowly 0.05, escape 0.01, attack 0.01, jump 0.00, 1314.8ms
# Speed: 85.3ms preprocess, 1314.8ms inference, 0.0ms postprocess per image at shape (1, 3, 1024, 1024)
# 文件slow_left0.png的类型是：缓慢左转，置信度是0.9269444346427917
# D:/digital_human/yolo_test_20240528/images_for_eval\slow_right0.png

# image 1/1 D:\digital_human\yolo_test_20240528\images_for_eval\slow_right0.png: 1024x1024 turnOnRightQuickly 0.92, escape 0.04, turnOnRightSlowly 0.02, ultraSkill 0.00, faceForwardMoveLeft 0.00, 1374.2ms
# Speed: 78.5ms preprocess, 1374.2ms inference, 0.0ms postprocess per image at shape (1, 3, 1024, 1024)
# 文件slow_right0.png的类型是：快速右转，置信度是0.9243348836898804
# D:/digital_human/yolo_test_20240528/images_for_eval\small_skill0.png

# image 1/1 D:\digital_human\yolo_test_20240528\images_for_eval\small_skill0.png: 1024x1024 smallSkill 0.89, attack 0.04, escape 0.03, ultraSkill 0.03, back 0.00, 1295.0ms
# Speed: 59.6ms preprocess, 1295.0ms inference, 0.0ms postprocess per image at shape (1, 3, 1024, 1024)
# 文件small_skill0.png的类型是：E技能，置信度是0.8853979110717773
# D:/digital_human/yolo_test_20240528/images_for_eval\speed_up0.png

# image 1/1 D:\digital_human\yolo_test_20240528\images_for_eval\speed_up0.png: 1024x1024 speedUp 0.69, back 0.20, escape 0.04, attack 0.04, ultraSkill 0.02, 1275.6ms
# Speed: 75.8ms preprocess, 1275.6ms inference, 0.0ms postprocess per image at shape (1, 3, 1024, 1024)
# 文件speed_up0.png的类型是：跑步，置信度是0.6850782036781311
# D:/digital_human/yolo_test_20240528/images_for_eval\ultra_skill0.png

# image 1/1 D:\digital_human\yolo_test_20240528\images_for_eval\ultra_skill0.png: 1024x1024 ultraSkill 1.00, back 0.00, smallSkill 0.00, faceForwardMoveLeft 0.00, escape 0.00, 1652.6ms
# Speed: 91.5ms preprocess, 1652.6ms inference, 0.0ms postprocess per image at shape (1, 3, 1024, 1024)
# 文件ultra_skill0.png的类型是：R技能，置信度是0.999894380569458