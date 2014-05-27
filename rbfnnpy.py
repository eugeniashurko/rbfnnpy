from numpy import array, append, vstack, transpose, reshape, \
                  dot, true_divide, mean, exp, sqrt, log, \
                  loadtxt, savetxt, zeros, frombuffer
from numpy.linalg import norm, lstsq
from multiprocessing import Process, Array
from random import sample
from time import time
from sys import stdout
from ctypes import c_double
from h5py import File


def metrics(a, b): 
    return norm(a - b)


def gaussian (x, mu, sigma): 
    return exp(- metrics(mu, x)**2 / (2 * sigma**2))


def multiQuadric (x, mu, sigma):
    return pow(metrics(mu,x)**2 + sigma**2, 0.5)


def invMultiQuadric (x, mu, sigma):
    return pow(metrics(mu,x)**2 + sigma**2, -0.5)


def plateSpine (x,mu):
    r = metrics(mu,x)
    return (r**2) * log(r)


class Rbf:
    def __init__(self, prefix = 'rbf', workers = 4, extra_neurons = 0, from_files = None):
        self.prefix = prefix
        self.workers = workers
        self.extra_neurons = extra_neurons

        # Import partial model
        if from_files is not None:            
            w_handle = self.w_handle = File(from_files['w'], 'r')
            mu_handle = self.mu_handle = File(from_files['mu'], 'r')
            sigma_handle = self.sigma_handle = File(from_files['sigma'], 'r')
            
            self.w = w_handle['w']
            self.mu = mu_handle['mu']
            self.sigmas = sigma_handle['sigmas']
            
            self.neurons = self.sigmas.shape[0]

    def _calculate_error(self, y):
        self.error = mean(abs(self.os - y))
        self.relative_error = true_divide(self.error, mean(y))

    def _generate_mu(self, x):
        n = self.n
        extra_neurons = self.extra_neurons

        # TODO: Make reusable
        mu_clusters = loadtxt('clusters100.txt', delimiter='\t')

        mu_indices = sample(range(n), extra_neurons)
        mu_new = x[mu_indices, :]
        mu = vstack((mu_clusters, mu_new))

        return mu

    def _calculate_sigmas(self):
        neurons = self.neurons
        mu = self.mu

        sigmas = zeros((neurons, ))
        for i in xrange(neurons):
            dists = [0 for _ in xrange(neurons)]
            for j in xrange(neurons):
                if i != j:
                    dists[j] = metrics(mu[i], mu[j])
            sigmas[i] = mean(dists)* 2
                      # max(dists) / sqrt(neurons * 2))
        return sigmas

    def _calculate_phi(self, x):
        C = self.workers
        neurons = self.neurons
        mu = self.mu
        sigmas = self.sigmas
        phi = self.phi = None
        n = self.n


        def heavy_lifting(c, phi):
            s = jobs[c][1] - jobs[c][0]
            for k, i in enumerate(xrange(jobs[c][0], jobs[c][1])):
                for j in xrange(neurons):
                    # phi[i, j] = metrics(x[i,:], mu[j])**3)
                    # phi[i, j] = plateSpine(x[i,:], mu[j]))
                    # phi[i, j] = invMultiQuadric(x[i,:], mu[j], sigmas[j]))
                    phi[i, j] = multiQuadric(x[i,:], mu[j], sigmas[j])
                    # phi[i, j] = gaussian(x[i,:], mu[j], sigmas[j]))
                if k % 1000 == 0:
                    percent = true_divide(k, s)*100
                    print c, ': {:2.2f}%'.format(percent)
            print c, ': Done'
        
        # distributing the work between 4 workers
        shared_array = Array(c_double, n * neurons)
        phi = frombuffer(shared_array.get_obj())
        phi = phi.reshape((n, neurons))

        jobs = []
        workers = []

        p = n / C
        m = n % C
        for c in range(C):
            jobs.append((c*p, (c+1)*p + (m if c == C-1 else 0)))
            worker = Process(target = heavy_lifting, args = (c, phi))
            workers.append(worker)
            worker.start()

        for worker in workers:
            worker.join()

        return phi

    def _do_algebra(self, y):
        phi = self.phi

        w = lstsq(phi, y)[0]
        os = dot(w, transpose(phi))
        return w, os
        # Saving to HDF5
        os_h5 = os_handle.create_dataset('os', data = os)

    def train(self, x, y):
        self.n = x.shape[0]

        ## Initialize HDF5 caches
        prefix = self.prefix
        postfix = str(self.n) + '-' + str(self.extra_neurons) + '.hdf5'
        name_template = prefix + '-{}-' + postfix
        phi_handle = self.phi_handle = File(name_template.format('phi'), 'w')
        os_handle = self.w_handle = File(name_template.format('os'), 'w')
        w_handle = self.w_handle = File(name_template.format('w'), 'w')
        mu_handle = self.mu_handle = File(name_template.format('mu'), 'w')
        sigma_handle = self.sigma_handle = File(name_template.format('sigma'), 'w')

        ## Mu generation
        mu = self.mu = self._generate_mu(x)
        self.neurons = mu.shape[0]
        print '({} neurons)'.format(self.neurons)
        # Save to HDF5
        mu_h5 = mu_handle.create_dataset('mu', data = mu)

        ## Sigma calculation
        print 'Calculating Sigma...'
        sigmas = self.sigmas = self._calculate_sigmas()
        # Save to HDF5
        sigmas_h5 = sigma_handle.create_dataset('sigmas', data = sigmas)
        print 'Done'

        ## Phi calculation
        print 'Calculating Phi...'
        phi = self.phi = self._calculate_phi(x)
        print 'Done'
        # Saving to HDF5
        print 'Serializing...'
        phi_h5 = phi_handle.create_dataset('phi', data = phi)
        del phi
        self.phi = phi_h5
        print 'Done'

        ## Algebra
        print 'Doing final algebra...'
        w, os = self.w, _ = self._do_algebra(y)
        # Saving to HDF5
        w_h5 = w_handle.create_dataset('w', data = w)
        os_h5 = os_handle.create_dataset('os', data = os)

        ## Calculate error
        self._calculate_error(y)
        print 'Done'

    def predict(self, test_data):
        mu = self.mu = self.mu.value
        sigmas = self.sigmas = self.sigmas.value
        w = self.w = self.w.value

        print 'Calculating phi for test data...'
        phi = self._calculate_phi(test_data)
        os = dot(w, transpose(phi))
        savetxt('iok3834.txt', os, delimiter='\n')
        return os

    @property
    def summary(self):
        return '\n'.join( \
            ['-----------------',
            'Training set size: {}'.format(self.n),
            'Hidden layer size: {}'.format(self.neurons),
            '-----------------',
            'Absolute error   : {:02.2f}'.format(self.error),
            'Relative error   : {:02.2f}%'.format(self.relative_error * 100)])


def predict(test_data):
    mu = File('rbf-mu-212243-2400.hdf5', 'r')['mu'].value
    sigmas = File('rbf-sigma-212243-2400.hdf5', 'r')['sigmas'].value
    w = File('rbf-w-212243-2400.hdf5', 'r')['w'].value

    n = test_data.shape[0]  
    neur = mu.shape[0]  
    
    mu = transpose(mu)
    mu.reshape((n, neur))   

    phi = zeros((n, neur)) 
    for i in range(n):
        for j in range(neur):
            phi[i, j] = multiQuadric(test_data[i,:], mu[j], sigmas[j])

    os = dot(w, transpose(phi))
    savetxt('iok3834.txt', os, delimiter='\n')
    return os