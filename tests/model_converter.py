import safetensors
import torch
 
# 假设你已经有了一个safetensors模型实例
# model: 模型实例
 
# 序列化模型权重到二进制文件
model = safetensors.safe_open(filename="D:/digital_human/distil-large-v3/model.safetensors") 
