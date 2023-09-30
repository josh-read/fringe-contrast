import numpy as np
import matplotlib.pyplot as plt

from fringe_contrast.beam import IdealBeam, BroadenedBeam
from fringe_contrast.interferometer import Interferometer


def main():
    times = np.linspace(-1, 3, 1001)
    ib = IdealBeam(532e-9)
    mi = Interferometer(ib, 1e-9)
    intensity_ib = [mi.detector_intensity(t) for t in times]
    bb = BroadenedBeam(532e-9, thickness=1e-7)
    mi = Interferometer(bb, 1e-9)
    intensity_bb = [mi.detector_intensity(t) for t in times]
    fig, ax = plt.subplots()
    ax.plot(times, intensity_ib)
    ax.plot(times, intensity_bb)


if __name__ == '__main__':
    main()
    plt.show()
