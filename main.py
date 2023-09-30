import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt


C = 3e8  # speed of light


class MichelsonInterferometer:

    def __init__(self, wavelength, tau):
        self.wavelength = wavelength
        self.frequency = C / wavelength
        self.angular_frequency = 2 * np.pi * self.frequency
        self.tau = tau

    def beam_field(self, t, phase):
        return np.cos(self.angular_frequency * t + phase)

    def detector_electric_field(self, t):
        pos_1 = self.position_history(t)
        pos_2 = self.position_history(t - self.tau)
        phase = 2 * (pos_1 - pos_2) / self.wavelength
        return lambda t_: self.beam_field(t_, 0) + self.beam_field(t_, phase)

    def detector_intensity(self, t):
        period = 1 / self.frequency
        res = integrate.quad(lambda t_: self.detector_electric_field(t)(t_)**2, 0, period)
        return res[0]

    def position_history(self, t):
        return 0 if t < 0 else 3e3 * t + 1e3 * t**2


def main():
    mi = MichelsonInterferometer(532e-9, 1e-9)
    times = np.linspace(-1, 3, 1001)
    intensity = [mi.detector_intensity(t) for t in times]
    fig, ax = plt.subplots()
    ax.plot(times, intensity)


if __name__ == '__main__':
    main()
    plt.show()
