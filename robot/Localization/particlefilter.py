import numpy as np
import scipy.stats
from numpy.random import uniform,randn,random
import matplotlib.pyplot as plt


def create_uniform_particles(x_range, y_range, hdg_range, N):
    particles = np.empty((N, 3))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = uniform(hdg_range[0], hdg_range[1], size=N)
    particles[:, 2] %= 2 * np.pi
    return particles

    
def predict(particles, u, std, dt=1.):
    """ move according to control input u (heading change, velocity)
    with noise Q (std heading change, std velocity)"""

    N = len(particles)
    # update heading
    particles[:, 2] += u[0] + (randn(N) * std[0])
    particles[:, 2] %= 2 * np.pi

    # move in the (noisy) commanded direction
    dist = (u[1] * dt) + (randn(N) * std[1])
    particles[:, 0] += np.cos(particles[:, 2]) * dist
    particles[:, 1] += np.sin(particles[:, 2]) * dist    
    
    
def update(particles, weights, z, R, landmarks):
    weights.fill(1.)
    for i, landmark in enumerate(landmarks):
        distance = np.linalg.norm(particles[:, 0:2] - landmark, axis=1)
        weights *= scipy.stats.norm(distance, R).pdf(z[i])

    weights += 1.e-300      # avoid round-off to zero
    weights /= sum(weights) # normalize    
    
    
def estimate(particles, weights):
    """returns mean and variance of the weighted particles"""

    pos = particles[:, 0:2]
    mean = np.average(pos, weights=weights, axis=0)
    var  = np.average((pos - mean)**2, weights=weights, axis=0)
    return mean, var    
    
    
def neff(weights):
    return 1. / np.sum(np.square(weights))    
    
    
def simple_resample(particles, weights):
    N = len(particles)
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1. # avoid round-off error
    indexes = np.searchsorted(cumulative_sum, random(N))

    # resample according to indexes
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights /= np.sum(weights) # normalize    

    
def run_pf(N, iters=18, sensor_std_err=0.1, xlim=(0, 20), ylim=(0, 20)):    
    landmarks = np.array([[-1, 2], [5, 10], [12,14], [18,21]]) 
    NL = len(landmarks)
    
    # create particles and weights
    particles = create_uniform_particles((0,20), (0,20), (0, 2*np.pi), N)
    weights = np.zeros(N)
    
    xs = []   # estimated values
    robot_pos = np.array([0., 0.])
    
    for x in range(iters):
        robot_pos += (1, 1) 
        
        # distance from robot to each landmark
        zs = np.linalg.norm(landmarks - robot_pos, axis=1) + (randn(NL) * sensor_std_err)
        
        # move particles forward to (x+1, x+1)
        predict(particles, u=(0.00, 1.414), std=(.2, .05))
        
        # incorporate measurements
        update(particles, weights, z=zs, R=sensor_std_err, landmarks=landmarks)
        
        # resample if too few effective particles
        if neff(weights) < N/2:
            simple_resample(particles, weights)
        
        # Computing the State Estimate
        mu, var = estimate(particles, weights)
        xs.append(mu)
    
    xs = np.array(xs)
    plt.plot(np.arange(iters+1),'k+')
    plt.plot(xs[:, 0], xs[:, 1],'r.')
    plt.scatter(landmarks[:,0],landmarks[:,1],alpha=0.4,marker='o',c=randn(4),s=100) # plot landmarks
    plt.legend( ['Actual','PF'], loc=4, numpoints=1)
    plt.xlim([-2,20])
    plt.ylim([0,22])
    print ('estimated position and variance:\n\t', mu, var)
    plt.show()
    
    
    
if __name__ == '__main__':    
    run_pf(N=5000)
