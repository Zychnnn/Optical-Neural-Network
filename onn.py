import torch
import numpy as np


def detector_region(x):
    return torch.cat((
        x[:, 46 : 66, 46 : 66].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 46 : 66, 93 : 113].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 46 : 66, 140 : 160].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 85 : 105, 46 : 66].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 85 : 105, 78 : 98].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 85 : 105, 109 : 129].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 85 : 105, 140 : 160].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 125 : 145, 46 : 66].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 125 : 145, 93 : 113].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 125 : 145, 140 : 160].mean(dim=(1, 2)).unsqueeze(-1)), dim=-1)


class DiffractiveLayer(torch.nn.Module):

    def __init__(self):
        super(DiffractiveLayer, self).__init__()
        self.size = 200                      # 200 * 200 neurons in one layer
        self.distance = 0.03                 # distance bewteen two layers (3cm)
        self.ll = 0.08                       # layer length (8cm)
        self.wl = 3e8 / 0.4e12               # wave length
        self.fi = 1 / self.ll                # frequency interval
        self.wn = 2 * 3.1415926 / self.wl    # wave number

        # self.phi (200, 200)
        phi = np.fromfunction(
            lambda x, y: np.square((x - (self.size // 2)) * self.fi) + np.square((y - (self.size // 2)) * self.fi),
            shape=(self.size, self.size)) # Removed dtype=np.complex64 to avoid warning

        # h (200, 200) - as a complex numpy array
        h_np = np.fft.fftshift(np.exp(1.0j * self.wn * self.distance) * np.exp(-1.0j * self.wl * np.pi * self.distance * phi))

        # Convert to a PyTorch complex tensor and register as a non-trainable buffer
        h_complex = torch.from_numpy(h_np.astype(np.complex64))
        self.register_buffer('h', h_complex, persistent=False)


    def forward(self, waves):
        # waves (batch, 200, 200, 2) -- input real tensor
        # 1. Convert real tensor to complex tensor
        waves_complex = torch.view_as_complex(waves)

        # 2. Perform 2D FFT using the modern API
        temp = torch.fft.fft2(waves_complex, dim=(1, 2))

        # 3. Perform complex multiplication in k-space (frequency domain)
        k_space = temp * self.h

        # 4. Perform inverse 2D FFT
        angular_spectrum_complex = torch.fft.ifft2(k_space, dim=(1, 2))
        
        # 5. Convert complex tensor back to real tensor for compatibility with the rest of the network
        angular_spectrum = torch.view_as_real(angular_spectrum_complex)

        return angular_spectrum


class Net(torch.nn.Module):
    """
    phase only modulation
    """
    def __init__(self, num_layers=5):

        super(Net, self).__init__()
        # self.phase (200, 200)
        self.phase = [torch.nn.Parameter(torch.from_numpy(2 * np.pi * np.random.random(size=(200, 200)).astype('float32'))) for _ in range(num_layers)]
        for i in range(num_layers):
            self.register_parameter("phase" + "_" + str(i), self.phase[i])
        self.diffractive_layers = torch.nn.ModuleList([DiffractiveLayer() for _ in range(num_layers)])
        self.last_diffractive_layer = DiffractiveLayer()
        self.sofmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        # x (batch, 200, 200, 2)
        for index, layer in enumerate(self.diffractive_layers):
            temp = layer(x)
            exp_j_phase = torch.stack((torch.cos(self.phase[index]), torch.sin(self.phase[index])), dim=-1)
            x_real = temp[..., 0] * exp_j_phase[..., 0] - temp[..., 1] * exp_j_phase[..., 1]
            x_imag = temp[..., 0] * exp_j_phase[..., 1] + temp[..., 1] * exp_j_phase[..., 0]
            x = torch.stack((x_real, x_imag), dim=-1)
        x = self.last_diffractive_layer(x)
        # x_abs (batch, 200, 200)
        x_abs = torch.sqrt(x[..., 0] * x[..., 0] + x[..., 1] * x[..., 1])
        output = self.sofmax(detector_region(x_abs))
        return output


if __name__ == '__main__':
    print(Net())