import numpy as np

class EMsim:
    """ This Class is used to simulate the motion of particles with charges under
    the influence of an electric field
    Attributes:
            phase-space
            t_start
            t_end
            t_step_base
            t
            boundary

    Methods:
            a_e_field
            a_e_particle
            t_step
            collisions
            update
            display

    """

    def __init__(self, phase_space, mass, charge, t_data = (0, 1, 0.1), boundary=False, fields = False):
        """Sets up the simulation

        Args:
            phase_space - a numpy matrix, where each row corresponds to the phase space
                          of each particle. e.g. [p1 p2 p3 ...] 
                          where p1 = [x1 y1 z1 vx1 vy1 vz1 q1 m1]
                          where ther 1st three are positions, followed by 3 velocities 
                          followed by the charge and mass of the particle


        """

        self.phase_space = phase_space
        self.t_start, self.t_end, self.t_step_base = t_data
        self.t = self.t_start
        self.boundary = boundary
        self.mass = mass
        self.charge = charge
        self.fields = fields

    def a_e_field(self, particle):
        pos = particle[:3]
        vel = particle[3:6]
        charge = particle[6]
        mass = particle[7]
        # check if the fields are varying
        if callable(self.fields):
            # calculate the fields( time dependent) and store only the e-field
            # wasteful in m-field calculation is long
            e_field = self.fields(self.t, pos)[0,:]

        return 0

    def a_e_particle(self, p_index):
        # k_e is the Coulomb's constant
        k_e = 8.99*(10**9)
        # indicies of other particles
        iop = np.arange(len(self.phase_space))
        # difference in positions of all particles with respect to the one being analyzed
        x_dif = self.phase_space[iop != p_index,0] - self.phase[p_index,0]
        y_dif = self.phase_space[iop != p_index,1] - self.phase[p_index,1]
        z_dif = self.phase_space[iop != p_index,2] - self.phase[p_index,2] 

        # an array of the cube of the absolute distance between the particles
        d_cube = (x_dif**2 + y_dif**2 + z_dif**2)**(3/2)
        # charge divided by the the distance cubed
        c_d_dc = self.charge[iop != p_index]/(d_cube)

        # charge/(mass* k_e)
        c_div_m = self.charge[p_index]/(k_e*self.mass[p_index])

        a_x = c_div_m*np.sum(x_dif*c_d_dc)
        a_y = c_div_m*np.sum(y_dif*c_d_dc)
        a_z = c_div_m*np.sum(z_dif*c_d_dc)

        return np.array((a_x,a_y,a_z))

    def t_step(self):

        max_v = np.amax(abs(self.phase_space[:,3:5]))
        t_step = self.t_step_base*(np.exp(-max_v))

        return t_step

    def collisions(self):
        return 0

    def update(self):
        return 0

    def display(self):
        return 0
