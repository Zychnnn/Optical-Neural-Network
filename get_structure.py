import torch
import numpy as np

# 假设您的onn.py文件在当前路径下
from onn import Net # 导入您的Net类

# 实例化模型
model = Net(num_layers=5) # 确保num_layers与您训练时一致
# 如果您的模型是在GPU上训练的，确保将其移动到CPU以便保存或在CPU上加载
model.load_state_dict(torch.load("./saved_model/136_model.pth", map_location=torch.device('cpu')))
model.eval() # 将模型设置为评估模式

# 获取每一层的相位图
all_phase_maps = []
# 遍历每一层
for i in range(model.num_layers):
    # 提取原始相位数据
    raw_phase = model.phase[i].data.cpu().numpy()
    
    # 对相位进行取模运算，将其约束在 [0, 2*pi) 的范围内
    # np.fmod 可以正确处理负数，结果符号与被除数相同
    # 但我们希望结果是正数，所以可以使用 `(raw_phase % (2 * np.pi) + (2 * np.pi)) % (2 * np.pi)`
    # 或者直接使用np.mod
    bounded_phase = np.mod(raw_phase, 2 * np.pi)
    
    all_phase_maps.append(bounded_phase)

# 示例：查看第一层修正后的相位图
print(f"修正后第一层相位图的形状: {all_phase_maps[0].shape}")
print(f"修正后第一层相位图的值范围: [{np.min(all_phase_maps[0])}, {np.max(all_phase_maps[0])}]")