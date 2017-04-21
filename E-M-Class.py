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
            b_field=np.array([0,0,0]), e_field=np.array([0,0,0]),
            accuracy=0.5, boundary=False
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
        self.e_field = e_field
        assert (accuracy <= 1) and (accuracy >= 0), 'Accuracy must be between [0,1]'
        self.accuracy = accuracy
        self.positions = [copy(phase_space[:,0:3])]

    @staticmethod
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

    @staticmethod
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
        field_comp = alpha[:,None] * (e_comp + cross_comp)
        # coulomb interaction components
        coul_comp = EMsim.coulomb_interactions(pos, c, m)
        return np.hstack((vel, field_comp + coul_comp))

    def t_step(self):

        max_v = np.amax(abs(self.phase_space[:,3:6]))
        t_step = self.t_step_base*(self.accuracy*np.exp(-max_v) + (1-self.accuracy))

        return t_step

    def collisions(self, t_step):
        if self.boundary:
            # check which particles are past the boundary and flip the corresponding velocity and move the particle back in the box
            for i, irow in enumerate(self.phase_space):
                # x component
                if irow[0] < self.boundary[0][0]:
                    self.phase_space[i,0] -= 2*self.boundary[0][0]
                    self.phase_space[i,3] *= -1
                else if irow[0] > self.boundary[0][1]:
                    self.phase_space[i,0] -= 2*self.boundary[0][1]
                    self.phase_space[i,3] *= -1
                #y component
                if irow[1] < self.boundary[1][0]:
                    self.phase_space[i,1] -= 2*self.boundary[1][0]
                    self.phase_space[i,4] *= -1
                else if irow[1] > self.boundary[1][1]:
                    self.phase_space[i,1] -= 2*self.boundary[1][1]
                    self.phase_space[i,4] *= -1
                # z component
                if irow[2] < self.boundary[2][0]:
                    self.phase_space[i,2] -= 2*self.boundary[2][0]
                    self.phase_space[i,5] *= -1
                else if irow[2] > self.boundary[2][1]:
                    self.phase_space[i,2] -= 2*self.boundary[2][1]
                    self.phase_space[i,5] *= -1


        # check for particle particle collisions
        for i, irow in enumerate(self.phase_space):
            row_idx = np.arange(self.phase_space.shape[0])
            for j, jrow in enumerate(self.phase_space[row_idx != i,:]):
                dif = irow - jrow
                same_position = (dif[0] <= 10**-16) and (dif[1] <= 10**-16) and (dif[2] <= 10**-16)
                crossed = (((dif[0] - dif[3]*t_step) <= 0) and
                            ((dif[1] - dif[4]*t_step) <= 0) and
                            ((dif[2] - dif[5]*t_step) <= 0))
                if same_position or crossed:
                    # Do particle collision correection




    def update(self):
        ts_tracker = 1

        while ((self.t_end-self.t) > (10**-16)):
            old_position_space = copy(self.phase_space[:,0:3])
            old_t = self.t
            t_step = self.t_step()
            self.t += t_step
            change = self.evolve(self.t, self.phase_space, self.mass, self.charge, self.b_field, self.e_field)
            self.phase_space += t_step*change
            self.collisions(t_step)
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
