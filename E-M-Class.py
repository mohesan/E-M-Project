import numpy as np
from copy import copy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

class EMsim:
    """ Electromagnetic Particle Interactions

    This class is used to simulate charged particle dynamics under the influence
    of electromagnetic fields and coulomb forces from neighbouring particles.

    Attributes:
        phase_space: particle position and velocity array (# particles, 6)
        t_start: starting time for simulation
        t_end: ending time for simulation
        t_step_base: desired time resolution
        t: current time
        boundary: rectangular boundary conditions

    Methods:
        coulomb_interactions
        t_step
        collisions
        update
        display

    """

    def __init__(
            self, phase_space, mass, charge, t_data = (0, 1, 0.1),
            boundary=False, b_field=False, e_field=False, accuracy=0.5
            ):
        """Sets up the simulation

        Args:
            phase_space - numpy ndarray, each row corresponds to the
                          phase space of each particle. e.g. [p1 p2 p3 ...]
                          where p1 = [x1 y1 z1 vx1 vy1 vz1]
            mass - ndarray, (# particles, ), particle masses
            charge - ndarray, (# particles, ), particle charges
            t_data - tuple, time data (start, end, time step)
            boundary - False if no boundary, otherwise tuple of 2-tuples
                       giving coordinate boundaries eg. ((0,1), (0,1), (0,1))
            b_field - function or ndarray with shape (3,), magnetic field
            e_field - function or ndarray with shape (3,), electric field
            accuracy - degree of time step adaptivity with regards to speed


        """

        self.phase_space = phase_space
        self.t_start, self.t_end, self.t_step_base = t_data
        self.t = self.t_start
        self.boundary = boundary
        self.mass = mass
        self.charge = charge
        self.b_field = b_field
        self.e_field = b_field
        assert (accuracy <= 1) and (accuracy >= 0), 'Accuracy must be between [0,1]'
        self.accuracy = accuracy
        self.positions = [copy(phase_space[:,0:3])]

    def coulomb_interactions(position, charge, mass):
        """Calculate coulomb acceleration contribution

        Take position, charge and mass information of a group of particles,
        and for each calculate the acceleration components due to the coulomb
        force from all other particles, returned as a ndarray.
        """
        # coulomb constant
        K_e = 8.99e9 # N m^2 C^-2
        # possible row indices
        row_idx = np.arange(position.shape[0])
        # prep the array to hold velocity derivs
        coul_accel = np.zeros_like(position)
        # iterate over particles, maybe this could be better
        for i, row in enumerate(position):
            c = charge[i] # charge of the particle under consideration
            c_other = charge[row_idx != i] # all other particle charges

            m = mass[i] # mass of particle under consideration
            m_other = mass[row_idx != i] # all other particle masses
            # position differences between particle and all others
            #2D array (# particles - 1, 3)
            p_diff = position[row_idx != i] - row

            d_cube = np.sum(p_diff**2, axis=1)**(3/2)

            c_div_m = c / (K_e*m) # float

            c_div_dc = c_other / d_cube # 1D array(# particles -1,)

            a = c_div_m * np.sum(c_div_dc[:,None] * p_diff, axis=0)

            coul_accel[i,:] = a
        return coul_accel

    def evolve(t, p, m, c, b, e):
        # charge - mass ratio
        alpha = c / m

        # split out the coordinates
        pos = p[:,:3]
        #x, y, z = p[:,0], p[:,1], p[:,2]
        # split out the velocities
        vel = p[:,3:]
        #vx, vy, vz = p[:,3], p[:,4], p[:,5]
        #vels = np.column_stack((vx,vy,vz))
        # is E varying
        if callable(e):
            e_comp = e(t, pos)
        else:
            e_comp = e
        # is B varying 
        if callable(b):
            b_comp = b(t, pos)
        else:
            b_comp = b
        # acceleration due to B-field
        cross_comp = np.cross(vel, b_comp)
        # acceleration from field interactions
        field_comp = alpha[:,None] * (E_comp + cross_comp)
        # coulomb interaction components
        coul_comp = coulomb_interactions(pos, charge, mass)
        return np.hstack((vels, field_comp + coul_comp))

    def a_e_particle(self, p_index):
        # k_e is the Coulomb's constant
        k_e = 8.99*(10**9)
        # indicies of other particles
        iop = np.arange(len(self.phase_space))
        # difference in positions of all particles with respect to the one being analyzed
        x_dif = self.phase_space[iop != p_index,0] - self.phase_space[p_index,0]
        y_dif = self.phase_space[iop != p_index,1] - self.phase_space[p_index,1]
        z_dif = self.phase_space[iop != p_index,2] - self.phase_space[p_index,2] 

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

        max_v = np.amax(abs(self.phase_space[:,3:6]))
        t_step = self.t_step_base*(self.accuracy*np.exp(-max_v) + (1-self.accuracy))

        return t_step

    def collisions(self):
        if self.boundary:
            # check which particles are past the boundary and flip the corresponding velocity and move the particle back in the box

        # check for particle particle collisions

    def update(self):
        ts_tracker = 0

        while ((self.t_end-self.t) < (10**-16)):
            old_position_space = copy(self.phase_space[:,0:3])
            old_t = self.t
            self.t += t_step()
            change = self.evolve(self.t, self.phase_space, self.mass, self.charge, self.b_field, self.e_field)
            self.phase_space += t_step*change
            self.collisions()
            if (self.t >= (ts_tracker*self.t_step_base)):
                weight = (ts_tracker*self.t_step_base - self.t)/(old_t-self.t)
                new_position_space = weight*old_position_space + (1-weight)*self.phase_space[:,0:3]
                self.positions.append(new_position_space)
                ts_tracker +=1


    def display(self):
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1], projection='3d')
        ax.view_init(30,0)
        ax.set_xlim((0,100))
        ax.set_ylim((0,100))
        ax.set_zlim(0, 10)
        ax.set_xlabel('Cheddar Gryphons')
        ax.set_ylabel('Parmesan Gryphons')
        ax.set_zlabel('Parmesan Unicorns')

        colors = plt.cm.jet(np.linspace(0,1,N_trajectories))
        pts = [ax.plot([],[],[],'o',c=c)[0] for c in colors]
        return 0
