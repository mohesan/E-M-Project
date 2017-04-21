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
            # 2D array (# particles - 1, 3)
            p_diff = position[row_idx != i] - row
            # 1D array (# particles - 1,)
            d_cube = np.sum(p_diff**2, axis=1)**(3/2)
            c_div_m = c / (K_e*m) # float
            c_div_dc = c_other / d_cube # 1D array (# particles -1,)
            # add an axis by slicing with None allowing broadcasting
            a = c_div_m * np.sum(c_div_dc[:,None] * p_diff, axis=0)
            coul_accel[i,:] = a
        return coul_accel

    @staticmethod
    def evolve(t, p, m, c, b, e):
        # charge - mass ratio
        alpha = c / m
        # split out the coordinates
        pos = p[:,:3]
        # split out the velocities
        vel = p[:,3:]
        # is e varying
        if callable(e):
            e_comp = e(t, pos)
        else:
            e_comp = e
        # is b varying
        if callable(b):
            b_comp = b(t, pos)
        else:
            b_comp = b
        # acceleration due to b-field
        cross_comp = np.cross(vel, b_comp)
        # acceleration from field interactions
        field_comp = alpha[:,None] * (e_comp + cross_comp)
        # coulomb interaction components
        coul_comp = EMsim.coulomb_interactions(pos, c, m)
        return np.hstack((vel, field_comp + coul_comp))

    def t_step(self):
        """Adaptive time step based on particle velocity"""
        max_v = np.amax(abs(self.phase_space[:,3:6]))
        t_step = self.t_step_base*(self.accuracy*np.exp(-max_v) + (1-self.accuracy))

        return t_step

    def collisions(self, t_step):
        if self.boundary:

            # check which particles are past the boundary and flip the
            # corresponding velocity and move the particle back in the box
            for i, irow in enumerate(self.phase_space):
                # x component
                if irow[0] < self.boundary[0][0]:
                    self.phase_space[i,0] = 2*self.boundary[0][0] - self.phase_space[i,0]
                    self.phase_space[i,3] *= -1
                elif irow[0] > self.boundary[0][1]:
                    self.phase_space[i,0] = 2*self.boundary[0][1] - self.phase_space[i,0]
                    self.phase_space[i,3] *= -1
                #y component
                if irow[1] < self.boundary[1][0]:
                    self.phase_space[i,1] = 2*self.boundary[1][0] - self.phase_space[i,1]
                    self.phase_space[i,4] *= -1
                elif irow[1] > self.boundary[1][1]:
                    self.phase_space[i,1] = 2*self.boundary[1][1] - self.phase_space[i,1]
                    self.phase_space[i,4] *= -1
                # z component
                if irow[2] < self.boundary[2][0]:
                    self.phase_space[i,2] = 2*self.boundary[2][0] - self.phase_space[i,2]
                    self.phase_space[i,5] *= -1
                elif irow[2] > self.boundary[2][1]:
                    self.phase_space[i,2] = 2*self.boundary[2][1] - self.phase_space[i,2]
                    self.phase_space[i,5] *= -1


        # check for particle particle collisions
        for i, irow in enumerate(self.phase_space):
            row_idx = np.arange(self.phase_space.shape[0])
            for j, jrow in enumerate(self.phase_space[row_idx != i,:]):
                dif = irow - jrow

                crossed = (((dif[0] - dif[3]*t_step) <= 0) and
                            ((dif[1] - dif[4]*t_step) <= 0) and
                            ((dif[2] - dif[5]*t_step) <= 0))
                if crossed:
                    p1 = copy(irow)
                    p2 = copy(jrow)
                    t_mass = self.mass[i] + self.mass[j]
                    d_mass = self.mass[i] - self.mass[j]
                    self.phase_space[i,3] = (p1[3] *(d_mass) + 2*self.mass[j]*p2[3])/t_mass
                    self.phase_space[j,3] = (p2[3] *(-d_mass) + 2*self.mass[i]*p1[3])/t_mass
                    self.phase_space[i,4] = (p1[4] *(d_mass) + 2*self.mass[j]*p2[4])/t_mass
                    self.phase_space[j,4] = (p2[4] *(-d_mass) + 2*self.mass[i]*p1[4])/t_mass
                    self.phase_space[i,5] = (p1[5] *(d_mass) + 2*self.mass[j]*p2[5])/t_mass
                    self.phase_space[j,5] = (p2[5] *(-d_mass) + 2*self.mass[i]*p1[5])/t_mass

                    self.phase_space[i,0] += (self.phase_space[i,3]-p1[3])*(t_step/2) 
                    self.phase_space[j,0] += (self.phase_space[j,3]-p2[3])*(t_step/2) 
                    self.phase_space[i,1] += (self.phase_space[i,4]-p1[4])*(t_step/2) 
                    self.phase_space[j,1] += (self.phase_space[j,4]-p2[4])*(t_step/2) 
                    self.phase_space[i,2] += (self.phase_space[i,5]-p1[5])*(t_step/2) 
                    self.phase_space[j,2] += (self.phase_space[j,5]-p2[5])*(t_step/2) 

                elif (dif[0] <= 10**-16) and (dif[1] <= 10**-16) and (dif[2] <= 10**-16):
                    p1 = copy(irow)
                    p2 = copy(jrow)
                    t_mass = self.mass[i] + self.mass[j]
                    d_mass = self.mass[i] - self.mass[j]
                    self.phase_space[i,3] = (p1[3] *(d_mass) + 2*self.mass[j]*p2[3])/t_mass
                    self.phase_space[j,3] = (p2[3] *(-d_mass) + 2*self.mass[i]*p1[3])/t_mass
                    self.phase_space[i,4] = (p1[4] *(d_mass) + 2*self.mass[j]*p2[4])/t_mass
                    self.phase_space[j,4] = (p2[4] *(-d_mass) + 2*self.mass[i]*p1[4])/t_mass
                    self.phase_space[i,5] = (p1[5] *(d_mass) + 2*self.mass[j]*p2[5])/t_mass
                    self.phase_space[j,5] = (p2[5] *(-d_mass) + 2*self.mass[i]*p1[5])/t_mass




    def update(self):
        ts_tracker = 1

        while ((self.t_end-self.t) > (1e-16)):
            old_position_space = copy(self.phase_space[:,:3])
            old_t = self.t
            t_step = self.t_step()
            self.t += t_step
            change = self.evolve(self.t, self.phase_space, self.mass,
                                 self.charge, self.b_field, self.e_field)
            self.phase_space = self.phase_space + t_step*change
            self.collisions(t_step)
            if (self.t >= (ts_tracker*self.t_step_base)):
                weight = (ts_tracker*self.t_step_base - self.t)/(-t_step)
                new_position_space = (weight*old_position_space +
                                      (1-weight)*self.phase_space[:,:3])
                self.positions.append(new_position_space)
                ts_tracker +=1




    def display(self):
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1], projection='3d')
        ax.view_init(30,0)

        poss = np.array(self.positions)
        states = [poss[:,i] for i in range(self.positions[0].shape[0])]

        if self.boundary:
            ax.set_xlim((self.boundary[0][0],self.boundary[0][1]))
            ax.set_ylim((self.boundary[1][0],self.boundary[1][1]))
            ax.set_zlim((self.boundary[2][0],self.boundary[2][1]))
        else:
            flattened_poss = np.vstack(poss)
            u_limits = np.amax(flattened_poss,axis=0)
            l_limits = np.amin(flattened_poss,axis=0)
            ax.set_xlim((l_limits[0],u_limits[0]))
            ax.set_ylim((l_limits[1],u_limits[1]))
            ax.set_zlim((l_limits[2],u_limits[2]))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
        colors = plt.cm.jet(np.linspace(0,1,self.positions[0].shape[0]))
        pts = [ax.plot([],[],[],'o',c=c)[0] for c in colors]
        def init():
            for pt in pts:
                pt.set_data([], [])
                pt.set_3d_properties([])
            return pts

        def animate(i, states, pts):
            for pt, state in zip(pts, states):
                pt.set_data(state[i,0], state[i,1])
                pt.set_3d_properties(state[i,2])
            ax.view_init(30, 0.3*i)
            fig.canvas.draw()
            return pts

        anim = animation.FuncAnimation(
                fig, animate, init_func=init, frames=1000,
                fargs=(states, pts)
                )
        return anim
