t = np.linspace(0, 100, 1000)
N_trajectories = 5
init_states = [
    [10,30,5],
    [5,40,10],
    [15,15,15],
    [10,60,5],
    [50,20,10]
]
states = np.asarray([odeint(gryphon_unicorn,
                            init_state,
                            t,
                            args=(gryphons_cheddar,
                                  gryphons_parmesan,
                                  unicorns_parmesan))
                            for init_state in init_states])

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
lines = [ax.plot([],[],[],'-',c=c)[0] for c in colors]
pts = [ax.plot([],[],[],'o',c=c)[0] for c in colors]

def init():
    for line, pt in zip(lines, pts):
        line.set_data([],[])
        line.set_3d_properties([])

        pt.set_data([], [])
        pt.set_3d_properties([])

    return lines + pts

def animate(i, states, lines, pts):
    for line, pt, state in zip(lines,pts, states):
        line.set_data(state[:i,0], state[:i, 1])
        line.set_3d_properties(state[:i, 2])

        pt.set_data(state[i, 0], state[i, 1])
        pt.set_3d_properties(state[i, 2])

    ax.view_init(30, 0.3*i)
    fig.canvas.draw()
    return lines + pts

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=1000, fargs=(states, lines, pts))
anim.save('gryp.mp4', fps=50)
