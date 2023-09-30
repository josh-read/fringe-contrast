import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt


C = 3e8  # speed of light


class InputBeam:

    def __init__(self, wavelength):
        self.wavelength = wavelength
        self.frequency = C / wavelength
        self.angular_frequency = 2 * np.pi * self.frequency

    def electric_field(self, t, phase):
        raise NotImplementedError


class IdealBeam(InputBeam):

    def electric_field(self, t, phase):
        return np.cos(self.angular_frequency * t + phase)


class BroadenedBeam(InputBeam):

    def electric_field(self, t, phase):
        def beam_field(x):
            additional_phase = 2 * np.pi * x / self.wavelength
            return self.reflectivity_vs_position(x) * np.cos(self.angular_frequency * t + phase - additional_phase)
        return integrate.quad(beam_field, 0, 100e-9)[0]

    @staticmethod
    def reflectivity_vs_position(x):
        """Area between 0<x<100e-9 is equal to 1."""
        return - 2e14 * x + 2e7


class MichelsonInterferometer:

    def __init__(self, input_beam, tau):
        self.input_beam = input_beam
        self.tau = tau

    def detector_electric_field(self, t):
        pos_1 = position_history(t)
        pos_2 = position_history(t - self.tau)
        phase = 2 * (pos_1 - pos_2) / self.input_beam.wavelength
        return lambda t_: self.input_beam.electric_field(t_, 0) + self.input_beam.electric_field(t_, phase)

    def detector_intensity(self, t):
        period = 1 / self.input_beam.frequency
        res = integrate.quad(lambda t_: self.detector_electric_field(t)(t_)**2, 0, period)
        return res[0]


def position_history(t):
    return 0 if t < 0 else 3e3 * t + 1e3 * t**2


def main():
    times = np.linspace(-1, 3, 1001)
    ib = IdealBeam(532e-9)
    mi = MichelsonInterferometer(ib, 1e-9)
    intensity_ib = [mi.detector_intensity(t) for t in times]
    bb = BroadenedBeam(532e-9)
    mi = MichelsonInterferometer(bb, 1e-9)
    intensity_bb = [mi.detector_intensity(t) for t in times]
    fig, ax = plt.subplots()
    ax.plot(times, intensity_ib)
    ax.plot(times, intensity_bb)


if __name__ == '__main__':
    main()
    plt.show()
