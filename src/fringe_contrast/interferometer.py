from scipy import integrate as integrate

from fringe_contrast.trajectories import slow_shock


class Interferometer:

    def __init__(self, input_beam, tau):
        self.input_beam = input_beam
        self.tau = tau

    def detector_electric_field(self, t):
        pos_1 = slow_shock(t)
        pos_2 = slow_shock(t - self.tau)
        phase = 2 * (pos_1 - pos_2) / self.input_beam.wavelength
        return lambda t_: self.input_beam.electric_field(t_, 0) + self.input_beam.electric_field(t_, phase)

    def detector_intensity(self, t):
        period = 1 / self.input_beam.frequency
        res = integrate.quad(lambda t_: self.detector_electric_field(t)(t_)**2, 0, period)
        return res[0]
