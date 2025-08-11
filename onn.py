import torch
import numpy as np
import torch.nn.functional as F

def detector_region(x):
    """
    提取探测器区域的光强信号。
    """
    return torch.cat((
        x[:, 46:66, 46:66].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 46:66, 93:113].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 46:66, 140:160].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 85:105, 46:66].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 85:105, 78:98].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 85:105, 109:129].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 85:105, 140:160].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 125:145, 46:66].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 125:145, 93:113].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 125:145, 140:160].mean(dim=(1, 2)).unsqueeze(-1)), dim=-1)


class DiffractiveLayer(torch.nn.Module):
    """
    模拟光在自由空间中的衍射传播。
    """
    def __init__(self):
        super(DiffractiveLayer, self).__init__()
        self.size = 200
        self.distance = 0.03
        self.ll = 0.08
        self.wl = 3e8 / 0.4e12
        self.fi = 1 / self.ll
        self.wn = 2 * np.pi / self.wl

        phi = np.fromfunction(
            lambda x, y: np.square((x - (self.size // 2)) * self.fi) + np.square((y - (self.size // 2)) * self.fi),
            shape=(self.size, self.size)
        )

        h_np = np.fft.fftshift(np.exp(1.0j * self.wn * self.distance) * np.exp(-1.0j * self.wl * np.pi * self.distance * phi))
        h_complex = torch.from_numpy(h_np.astype(np.complex64))
        self.register_buffer('h', h_complex, persistent=False)

    def forward(self, waves):
        waves_complex = torch.view_as_complex(waves)
        temp = torch.fft.fft2(waves_complex, dim=(1, 2))
        k_space = temp * self.h
        angular_spectrum_complex = torch.fft.ifft2(k_space, dim=(1, 2))
        angular_spectrum = torch.view_as_real(angular_spectrum_complex)
        return angular_spectrum


class Net(torch.nn.Module):
    """
    全光衍射神经网络，包含多个相位调制层。
    """
    def __init__(self, num_layers=5):
        super(Net, self).__init__()
        # 使用 torch.nn.ParameterList 管理相位参数
        self.phase = torch.nn.ParameterList(
            [torch.nn.Parameter(2 * np.pi * torch.rand(size=(200, 200), dtype=torch.float32)) for _ in range(num_layers)]
        )
        self.diffractive_layers = torch.nn.ModuleList([DiffractiveLayer() for _ in range(num_layers)])
        self.last_diffractive_layer = DiffractiveLayer()
        
    def forward(self, x):
        for index, layer in enumerate(self.diffractive_layers):
            temp = layer(x)

            # 关键修改：移除对相位参数的取模操作，让优化器自由更新。
            # torch.sin和torch.cos的周期性保证了物理效果的一致性，同时不截断梯度。
            exp_j_phase = torch.stack((torch.cos(self.phase[index]), torch.sin(self.phase[index])), dim=-1)

            x_real = temp[..., 0] * exp_j_phase[..., 0] - temp[..., 1] * exp_j_phase[..., 1]
            x_imag = temp[..., 0] * exp_j_phase[..., 1] + temp[..., 1] * exp_j_phase[..., 0]
            x = torch.stack((x_real, x_imag), dim=-1)

        x = self.last_diffractive_layer(x)

        # 使用光强进行分类，这是正确的处理方式
        intensity = x[..., 0]**2 + x[..., 1]**2
        
        output = detector_region(intensity)
        
        return output

if __name__ == '__main__':
    print(Net())