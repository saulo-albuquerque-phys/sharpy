import sharpy
from sharpy.GW_likelihood import template_mlgw_bns
import jax
import jax.numpy as jnp

truth =  jnp.array([3.0, 1.0, 5.5, jnp.pi/2, jnp.pi, jnp.pi/2, 30.0, 0.7, 0.0, -0.1, 0.1,50.,500.])
psd = sharpy.PSDs.__path__[0] + "/LIGO-P1200087-v18-aLIGO_DESIGN_psd.dat"

detector_settings = {
        "H1": {
            "psd_file"  : psd,

            
        },
        "L1": {
            "psd_file"  : psd, 
        },}

from sharpy.GW_likelihood import GWNetwork, log_likelihood_det_mlgw_bns
truth =  jnp.array([3.0, 1.0,6., jnp.pi/2, jnp.pi, jnp.pi/2, 30.0, 0.7, 0.0, -0.1, 0.1,50.,500.])
gw_network = GWNetwork(detector_settings,
                            injection_parameters=truth,
                      )


from functools import partial
batched_detector        = gw_network.batched_detector
log_likelihood          = partial(log_likelihood_det_mlgw_bns, detector_list=batched_detector)

prior_bounds            = jnp.array([[0., 2*jnp.pi], [-jnp.pi/2, jnp.pi/2], [4.9, 8.7], [0., jnp.pi], [0., 2*jnp.pi], [0., jnp.pi], [25, 35], [0.4, 1.], [-1e-1, 1e-1], [-1., 1.], [-1., 1.],[5,1000],[5,1000]])
boundary_conditions     = jnp.array([1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,0,0]) #1: periodic, 0: reflective

number_of_particles     = 100
step_size               = 0.2
alpha                   = 0.9
id=4

parameters_names        =  ['ra','dec','logdistance','theta_jn','phiref','pol', 'mc','q', 'tc', 'chi1', 'chi2','lambda_1', 'lambda_2']
folder                  = "test_GW170817"
folder                  = f"{folder}/run_{id}"
label                   = f"run_{id}"


import os

if not os.path.exists(folder):
    os.makedirs(folder)

def prior(params):
    return 0.

from sharpy.smc_functions import run_sharpy

result_dict = run_sharpy(log_likelihood, 
                    prior, 
                    prior_bounds,
                    boundary_conditions, 
                    alpha,
                    number_of_particles, 
                    step_size,   
                    jax.random.PRNGKey(42),
                    folder = ".",
                    label = "run",
                    initial_particles = "prior",)

import numpy
samples     = result_dict['posterior_samples']

from corner import corner
fig = corner(numpy.array(samples), 
            show_titles    =True,
            labels          = parameters_names, 
             truths= truth,
            title_kwargs   = {"fontsize": 12},range=prior_bounds)

fig.savefig(f"{folder}/{label}_corner.png")
plt.show()
