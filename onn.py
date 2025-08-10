import torch
import numpy as np

def detector_region(x):
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
        x[:, 125:145, 140:160].mean(dim=(1, 2)).unsqueeze(-1)
    ), dim=-1)


class DiffractiveLayer(torch.nn.Module):
    def __init__(self):
        super(DiffractiveLayer, self).__init__()
        self.size = 200                 # 200 × 200 neurons in one layer
        self.distance = 0.03             # distance between two layers (3 cm)
        self.ll = 0.08                   # layer length (8 cm)
        self.wl = 3e8 / 0.4e12            # wavelength
        self.fi = 1 / self.ll             # frequency interval
        self.wn = 2 * np.pi / self.wl     # wave number

        # φ (200, 200)
        phi = np.fromfunction(
            lambda x, y: np.square((x - (self.size // 2)) * self.fi) +
                         np.square((y - (self.size // 2)) * self.fi),
            shape=(self.size, self.size)
        )

        # h (200, 200) - complex transfer function
        h_np = np.fft.fftshift(
            np.exp(1.0j * self.wn * self.distance) *
            np.exp(-1.0j * self.wl * np.pi * self.distance * phi)
        )

        # Convert to PyTorch complex tensor
        h_complex = torch.from_numpy(h_np.astype(np.complex64))
        self.register_buffer('h', h_complex, persistent=False)

    def forward(self, waves):
        # waves: (batch, 200, 200, 2) real tensor
        waves_complex = torch.view_as_complex(waves)

        # FFT → multiply in k-space → iFFT
        temp = torch.fft.fft2(waves_complex, dim=(1, 2))
        k_space = temp * self.h
        angular_spectrum_complex = torch.fft.ifft2(k_space, dim=(1, 2))

        return torch.view_as_real(angular_spectrum_complex)


class Net(torch.nn.Module):
    """ Phase-only modulation ONN """

    def __init__(self, num_layers=5):
        super(Net, self).__init__()
        self.phase = [
            torch.nn.Parameter(
                torch.from_numpy(2 * np.pi * np.random.random(size=(200, 200)).astype('float32'))
            ) for _ in range(num_layers)
        ]
        for i in range(num_layers):
            self.register_parameter(f"phase_{i}", self.phase[i])

        self.diffractive_layers = torch.nn.ModuleList([DiffractiveLayer() for _ in range(num_layers)])
        self.last_diffractive_layer = DiffractiveLayer()

    def forward(self, x):
        for index, layer in enumerate(self.diffractive_layers):
            temp = layer(x)

            # Phase constraint to [0, 2π]
            bounded_phase = self.phase[index] % (2 * np.pi)
            exp_j_phase = torch.stack((torch.cos(bounded_phase), torch.sin(bounded_phase)), dim=-1)

            # Apply phase modulation
            x_real = temp[..., 0] * exp_j_phase[..., 0] - temp[..., 1] * exp_j_phase[..., 1]
            x_imag = temp[..., 0] * exp_j_phase[..., 1] + temp[..., 1] * exp_j_phase[..., 0]
            x = torch.stack((x_real, x_imag), dim=-1)

        x = self.last_diffractive_layer(x)

        # Convert to intensity (|E|^2)
        intensity = x[..., 0] ** 2 + x[..., 1] ** 2

        # Detector readout
        output = detector_region(intensity)

        return output


if __name__ == '__main__':
    print(Net())
