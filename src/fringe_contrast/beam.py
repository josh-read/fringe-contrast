import numpy as np
from scipy import integrate as integrate


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

    def __init__(self, *args, thickness=100e-9, **kwargs):
        self.thickness = thickness
        self.intercept = 2 / thickness
        self.gradient = self.intercept / thickness
        super(BroadenedBeam, self).__init__(*args, **kwargs)

    def electric_field(self, t, phase):
        def beam_field(x):
            additional_phase = 2 * np.pi * x / self.wavelength
            return self.reflectivity_vs_position(x) * np.cos(self.angular_frequency * t + phase - additional_phase)
        return integrate.quad(beam_field, 0, 100e-9)[0]

    def reflectivity_vs_position(self, x):
        """Area between 0<x<thickness is equal to 1."""
        return - self.gradient * x + self.intercept

