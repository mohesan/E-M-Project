import numpy as np

class EMsim:
    """ This Class is used to simulate the motion of particles with charges under
    the influence of an electric field
    Attributes:

    Methods:


    """

    def __init__(self, phase_space, t_data = (0, 1, 0.1), boundary=False, fields = False):
        """Sets up the simulation

        Args:

        """

        self.phase_space = phase_space
        self.t_start, self.t_end, self.t_step_base = t_data
        self.boundary = boundary

    def a_e_field(self):
        return 0

    def a_e_particle(self):
        return 0

    def max_v(self):
        return 0

    def collisions(self):
        return 0

    def update(self):
        return 0

    def display(self):
        return 0
