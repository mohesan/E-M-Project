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
        t_start:     starting time for simulation (float)
        t_end:       ending time for simulation (float)
        t_step_base: desired time resolution (float)
        t:           current time (float)
        boundary:    False if no boundary, otherwise tuple of 2-tuples
                     giving coordinate boundaries eg. ((0,1), (0,1), (0,1))
        mass:        mass of all particles array of floats, (# particles, 1)
        charge:      charge of all particles array of floats, (# particles, 1)
        b_field:     magnetic field, function or array of floats with shape (3,)
        e_field:     electric field, function or array of floats with shape (3,)
        accuracy:    degree of time step adaptivity with regards to speed float in bounds of [0,1]
        positions:   the positions of all the particles at times t=t_step_base*i where i >=0 is an integer
                     and t_start <= t <= t_end a 3D array of floats (int((t_end-t_start)/t_step_base),# particles,3)
        animation:   Animation of the particles between t_start and t_end as created by create_animation()
                     (matplotlib.animation instance)
        optimal_fps: The optimal frames per second to view the animation so that one second in the animation is 
                     one second in real time (int)

    Methods:
        coulomb_interactions: Calculates the acceleration caused by all particles on all other particles
                              (Returns array (#particles, 3))
        evolve:               Calculates the new differentials i.e. new velocities and new accelerations
                              (Returns array (#particles, 6))
        t_step:               Calculates the time_step according to the maximum speed of the particles
                              and the accuracy attribute (Returns float)
        collisions:           Corrects the phase_space attribute in case of collisions between a particle
                              and the boundary or between particles (Returns pass)
        rk4:                  Use forth order Runge-Kutta method to determine the evolution of the 
                              phase-space (Returns pass)
        update:               Runs the simulation until t_end (Returns pass)
        save_animation:       Saves the animation instance as an mpeg file (Returns pass)
        create_animation:     Creates the animation passed on the positions attribute (Returns pass)

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
            mass -------- ndarray, (# particles, 1), particle masses
            charge ------ ndarray, (# particles, 1), particle charges
            t_data ------ tuple, time data (start, end, time step)
            b_field ----- function or ndarray with shape (3,1), magnetic field
            e_field ----- function or ndarray with shape (3,1), electric field
            accuracy ---- float in bounds of [0,1], degree of time step adaptivity 
                          with regards to speed
            boundary ---- False if no boundary, otherwise tuple of 2-tuples
                          giving coordinate boundaries eg. ((0,1), (0,1), (0,1))
        """
        # Validating input as well as initializing the class
        assert (type(phase_space) is np.ndarray) and (phase_space.shape[1] ==6), \
            'The phase space must only contain 3 positions and 3 velocities per particle and must be of type numpy.ndarray'
        self.phase_space = phase_space
        assert (t_data[1] > t_data[0]), 'end time provided must be greater than start time'
        assert((t_data[1]-t_data[0]) > t_data[2]), 'the base time step must be smaller then the time interval'
        assert(t_data[2] > 0), 'the base time step must be greater than 0'
        self.t_start, self.t_end, self.t_step_base = t_data
        self.t = self.t_start
        if boundary:
            assert (boundary[0][0] < boundary[0][1]) and (boundary[1][0] < boundary[1][1]) and (boundary[2][0] < boundary[2][1]),\
                'Please check that the tuple for boundaries have the lower limit first'
        self.boundary = boundary
        assert(type(mass) is np.ndarray) , 'The mass must be of type numpy.ndarray'
        assert ((mass.shape[0])==phase_space.shape[0]) and (len(mass.shape)==1), 'Please check that the size of the mass array corresponds to the phase space'
        self.mass = mass
        assert(type(charge) is np.ndarray) , 'The charge must be of type numpy.ndarray'
        assert ((charge.shape[0])==phase_space.shape[0]) and (len(charge.shape)==1), 'Please check that the size of the charge array corresponds to the phase space'
        self.charge = charge
        assert (callable(b_field)) or (b_field.shape == (3,)), 'Please check that the b_field is either a function or an array of size (3,)'
        self.b_field = b_field
        assert (callable(e_field)) or (e_field.shape == (3,)), 'Please check that the e_field is either a function or an array of size (3,)'
        self.e_field = e_field
        assert (accuracy <= 1) and (accuracy >= 0), 'Accuracy must be between [0,1]'
        self.accuracy = accuracy
        # Copy the phase_space so that they aren't linked
        self.positions = [copy(phase_space[:,0:3])]
        # Once the method create_animation() is run it will set the animation instance to this attribute
        self.animation = None
        # The optimal_fps is used when saving the animation instance as an mpeg
        # The optimal fps is the number of frames per second needed so that the mpeg runs at the same rate as real time
        self.optimal_fps = None

    @staticmethod
    def coulomb_interactions(position, charge, mass):
        """Calculate coulomb acceleration contribution

        Take position, charge and mass information of a group of particles,
        and for each calculate the acceleration components due to the coulomb
        force from all other particles.

        Args:
            position: 2D ndarray, shape (# particles, 3), columns are x,y,z
            charge: 1D ndarray, shape (# particles,)
            mass: 1D ndarray, shape (# particles,)

        Return:
            coul_accel: 2D ndarray, shape (# particles, 3), columns are
                        acceleration components x,y,z respectivel
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
            p_diff = row - position[row_idx != i]
            # 1D array (# particles - 1,)
            d_cube = np.sum(p_diff**2, axis=1)**(3/2)
            c_div_m = c*K_e / m # float
            c_div_dc = c_other / d_cube # 1D array (# particles -1,)
            # add an axis by slicing with None allowing broadcasting
            coul_accel[i,:] = c_div_m * np.sum(c_div_dc[:,None] * p_diff, axis=0)
        return coul_accel

    @staticmethod
    def evolve(t, p, m, c, b, e):
        """Phase space evolution

        Propagate the phase space on time step forward.

        Args:
            t: float, the time value
            p: 2D ndarray, shape (# particles, 6), columns correspond to
               x,y,z positions and velocities respectively
            m: 1D ndarray, shape (# particles,), masses
            c: 1D ndarray, shape (# particles,), charges
            b: function or 1D ndarry, magnetic field
            e: function or 1D ndarray, electric field

        Return:
            phase_deriv: 2D ndarray, shape (# particles, 6), rate of change of
                         phase space components

        """
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
        # magnetic field interaction
        # 2D ndarray, shape (# particles, 3)
        cross_comp = np.cross(vel, b_comp)
        # acceleration due to field interactions
        # use slicing with None to add extra dimension to alpha
        # allowing broadcasting
        field_comp = alpha[:,None] * (e_comp + cross_comp)
        # coulomb interaction components
        coul_comp = EMsim.coulomb_interactions(pos, c, m)
        phase_deriv = np.hstack((vel, field_comp + coul_comp))
        return phase_deriv

    def t_step(self):
        """Adaptive time step based on particle velocity"""
        # find the max velocity on the group of paricles
        max_v = np.amax(abs(self.phase_space[:,3:6]))
        # scale the time step depending on the speed
        # smaller time steps when the particles are moving fast allows
        # more precise computation
        t_step = self.t_step_base*(self.accuracy*np.exp(-max_v) + (1-self.accuracy))
        return t_step

    def collisions(self, t_step):
        """Checks to see if any of the particles has collided with the boundary or with another particle. 
           If collision has occured than an approximate correction will occur. Assumed that all 
           collisions are elastic in nature"""

        # check for particle particle collisions
        for i, irow in enumerate(self.phase_space):
            row_idx = np.arange(self.phase_space.shape[0])
            for j, jrow in enumerate(self.phase_space[row_idx != i,:]):
                # The difference in phase space
                dif = irow - jrow

                # two particles cross paths if the (x1-x2)*(x1' - x2') <= 0 for each component (x,y,z)
                # In some cases this algorithm will have false-positives but that is the inaccuracy associated
                # with the size of the time_step. smalled timesteps = more accuracy
                crossed = (((dif[0]*(dif[0] - dif[3]*t_step)) <= 0) and
                            ((dif[1]*(dif[1] - dif[4]*t_step)) <= 0) and
                            ((dif[2]*(dif[2] - dif[5]*t_step)) <= 0))
                if crossed:
                    # deep copy of both particles phase space coordinates
                    p1 = copy(irow)
                    p2 = copy(jrow)
                    # total mass
                    t_mass = self.mass[i] + self.mass[j]
                    #difference in their masses
                    d_mass = self.mass[i] - self.mass[j]
                    # Correction in velocities throuh conservation of momentum and kinetic energy
                    self.phase_space[i,3:] = (p1[3:] *(d_mass) + 2*self.mass[j]*p2[3:])/t_mass
                    self.phase_space[j,3:] = (p2[3:] *(-d_mass) + 2*self.mass[i]*p1[3:])/t_mass
                    # Correction in position is esitmated by calculating the displacement,d, as
                    # d = v_old*(2/5)*timestep + v_new*(3/5)*timestep 
                    # The new velocities are preffered to stop particles from passing through each other
                    # in some cases it is still possible for this 'tunnerling' to occur
                    self.phase_space[i,:3] += (self.phase_space[i,3:]-p1[3:])*(3*t_step/5) 
                    self.phase_space[j,:3] += (self.phase_space[j,3:]-p2[3:])*(3*t_step/5)

                # If the particle's path didnt cross with another particle then check to see that the particle's path 
                # is not currently at the same position as another particle
                elif (dif[0] <= 10**-16) and (dif[1] <= 10**-16) and (dif[2] <= 10**-16):
                    p1 = copy(irow)
                    p2 = copy(jrow)
                    t_mass = self.mass[i] + self.mass[j]
                    d_mass = self.mass[i] - self.mass[j]
                    self.phase_space[i,3:] = (p1[3:] *(d_mass) + 2*self.mass[j]*p2[3:])/t_mass
                    self.phase_space[j,3:] = (p2[3:] *(-d_mass) + 2*self.mass[i]*p1[3:])/t_mass

        # Only need to check for boundary collisions if a boundary is defined   
        if self.boundary:
            # Check which particles are past the boundary and flip the
            # Corresponding velocity and move the particle back in the box
            # The while loop is to help solve the case when velocities are very large.

            # Number of boundary corrections
            n_b_c = 1
            while n_b_c > 0 :
                n_b_c = 0
                for i, irow in enumerate(self.phase_space):
                    # x component of the particle is past the lower boundary
                    if irow[0] < self.boundary[0][0]:
                        # Move the particle back in the box
                        self.phase_space[i,0] = 2*self.boundary[0][0] - self.phase_space[i,0]
                        # Reflect the corresponding velocity
                        self.phase_space[i,3] *= -1
                        n_b_c +=1
                    # x component of the particle is past the upper boundary
                    elif irow[0] > self.boundary[0][1]:
                        self.phase_space[i,0] = 2*self.boundary[0][1] - self.phase_space[i,0]
                        self.phase_space[i,3] *= -1
                        n_b_c +=1
                    # y component of the particle is past the lower boundary
                    if irow[1] < self.boundary[1][0]:
                        self.phase_space[i,1] = 2*self.boundary[1][0] - self.phase_space[i,1]
                        self.phase_space[i,4] *= -1
                        n_b_c +=1
                    # y component of the particle is past the upper boundary
                    elif irow[1] > self.boundary[1][1]:
                        self.phase_space[i,1] = 2*self.boundary[1][1] - self.phase_space[i,1]
                        self.phase_space[i,4] *= -1
                        n_b_c +=1
                    # z component of the particle is past the lower boundary
                    if irow[2] < self.boundary[2][0]:
                        self.phase_space[i,2] = 2*self.boundary[2][0] - self.phase_space[i,2]
                        self.phase_space[i,5] *= -1
                        n_b_c +=1
                    # z component of the particle is past the upper boundary
                    elif irow[2] > self.boundary[2][1]:
                        self.phase_space[i,2] = 2*self.boundary[2][1] - self.phase_space[i,2]
                        self.phase_space[i,5] *= -1
                        n_b_c +=1




    def rk4(self, t_step):
        """Runge-Kutta 4th Order Integration

        Use the evolve method and a supplied time step to advance the
        phase space attribute through time.

        """
        # first order
        k1 = ((self.evolve(self.t, self.phase_space, self.mass,
                           self.charge, self.b_field, self.e_field))
                           *t_step)

        # half step, first order deriv
        xk = self.phase_space + k1*0.5
        tk = self.t + t_step*0.5

        # second order
        k2 = ((self.evolve(tk, xk, self.mass, self.charge,
                           self.b_field, self.e_field))
                           *t_step)

        # half step, second order deriv
        xk = self.phase_space + k2*0.5
        # third order
        k3 = ((self.evolve(tk, xk, self.mass, self.charge,
                           self.b_field, self.e_field))
                           *t_step)

        # full step, third order deriv
        xk = self.phase_space + k3
        tk = self.t + t_step
        # fourth order
        k4 = ((self.evolve(tk, xk, self.mass, self.charge,
                           self.b_field, self.e_field))
                           *t_step)

        # full runge-kutta phase step
        self.phase_space = self.phase_space + (k1 +2*(k2+k3) +k4)/6

    def update(self):
        """Move the particles through the entire time range"""
        # how many time steps have we done
        ts_tracker = 1
        # while we're not at the end
        while ((self.t_end-self.t) > (1e-16)):
            old_position_space = copy(self.phase_space[:,:3])
            old_t = self.t
            t_step = self.t_step()
            self.rk4(t_step)
            self.t += t_step
            self.collisions(t_step)
            # filter out when we are at integer multiples of the base time step
            if (self.t >= (ts_tracker*self.t_step_base)):
                # weighted smoothing to calculate the phase space
                # between time steps
                weight = (ts_tracker*self.t_step_base - self.t)/(-t_step)
                new_position_space = (weight*old_position_space +
                                      (1-weight)*self.phase_space[:,:3])
                self.positions.append(new_position_space)
                ts_tracker +=1

    def save_animation(self, name, fps=False):
        """Write animation to supplied file path

        This function can be run after using create_animation. If supplied with
        only the name parameter, it will use the optimal fps to ensure that
        the animation speed is 1 second simulation time per real time second.

        Args:
            name: a qualified, writable, file path to store the mp4 animation
            fps: False if optimal_fps desired, otherwise positive integer
        """
        if not fps:
            fps = self.optimal_fps

        self.animation.save(name, fps=fps)

    def create_animation(self):
        """Animate movement of particles through 3-space"""
        # reshape positions into appropriate format
        # each row of state is an entire particle trajectory
        poss = np.array(self.positions)
        states = [poss[:,i] for i in range(self.positions[0].shape[0])]
        # set up figure and axes
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1], projection='3d')
        ax.view_init(30,0)
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
        # map colours to charge
        cmap = plt.cm.get_cmap('seismic')
        colors = cmap((self.charge / 2*np.amax(self.charge) + 0.5).astype(int)*255)
        pts = [ax.plot([],[],[],'o',c=c)[0] for c in colors]

        # animation initialization function
        def init():
            for pt in pts:
                pt.set_data([], [])
                pt.set_3d_properties([])
            return pts

        # animating done here
        def animate(i, states, pts):
            for pt, state in zip(pts, states):
                pt.set_data(state[i,0], state[i,1])
                pt.set_3d_properties(state[i,2])
            ax.view_init(30, 0.3*i)
            fig.canvas.draw()
            return pts

        # ensure frames cover full trajectory
        num_frames = states[0].shape[0]
        anim = animation.FuncAnimation(
                fig, animate, init_func=init, frames=num_frames,
                fargs=(states, pts)
                )
        self.animation=anim
        # fps to use if animation should flow at 1 real time second
        self.optimal_fps = int(1/self.t_step_base)
