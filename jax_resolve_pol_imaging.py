import jax
from jax import numpy as jnp

import nifty8.re as jft
import resolve as rve
import resolve.re as jrve
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from resolve.re.sky_model import build_cf

import configparser
from jax import random
from src.stokes_adder import StokesAdder


# choose between ducc0 and finufft backend
response = 'ducc0'
#response = 'finufft'


seed = 42
key = random.PRNGKey(seed)
jax.config.update("jax_enable_x64", True)

obs = rve.ms2observations("Data/0415+379.u.2021_04_09.ms", "DATA", True, 0, "all")[0]
obs = obs.to_double_precision()
cfg = configparser.ConfigParser()
cfg.read("pol_imaging_only.cfg")

sky_dom = rve.sky_model._spatial_dom(cfg["sky"])
bg_shape = sky_dom.shape
bg_distances = sky_dom.distances

prefix_list = ['stokesI diffuse space i0', 'stokesQ diffuse space i0', 
               'stokesU diffuse space i0', 'stokesV diffuse space i0']

Stokes_list = []

for ii in range(4):
    bg_log_diff, additional = build_cf(prefix_list[ii], cfg["sky"], bg_shape, bg_distances)
    Stokes_list.append(bg_log_diff)


Stokes_dict = {"a": Stokes_list[0], "b": Stokes_list[1], 
               "c": Stokes_list[2], "d": Stokes_list[3]}
Pol_sky = StokesAdder(Stokes_dict)


key ,subkey = random.split(key)
mock_position = jft.Vector(jft.random_like(subkey, Pol_sky.domain))
pol_sky_prior = Pol_sky(mock_position)

# plot sky prior
'''
for ii in range(4):
    plt.imshow(pol_sky_prior[ii,:,:])
    plt.show()
'''

sdom = rve.sky_model._spatial_dom(cfg["sky"])
pdom = rve.PolarizationSpace(["I", "Q", "U", "V"])
sky_dom = rve.default_sky_domain(sdom=sdom, pdom=pdom)

sky_domain_dict = dict(npix_x=sdom.shape[0],
                       npix_y=sdom.shape[1],
                       pixsize_x=sdom.distances[0],
                       pixsize_y=sdom.distances[1],
                       pol_labels=['I','Q','U','V'],
                       times=[0.],
                       freqs=[0.])

R_new = jrve.InterferometryResponse(obs, sky_domain_dict, False, 1e-9, backend=response)
signal_response = lambda x: R_new(Pol_sky(x))

nll = jft.Gaussian(obs.vis.val, obs.weight.val).amend(signal_response)


n_vi_iterations = 15
delta = 1e-8
absdelta = delta * jnp.prod(jnp.array(sdom.shape))
n_samples = 2
odir = f"pol_imaging_results_jax_resolve"

def callback(samples, opt_state):
    if n_samples == 0:
        post_sr_mean = Pol_sky(samples)
    else:
        post_sr_mean = jft.mean(tuple(Pol_sky(s) for s in samples))
    plt.imshow(post_sr_mean[0, 0, 0, :, :].T, origin="lower", norm=LogNorm())
    plt.colorbar()
    plt.savefig(f"{odir}/StokesI_sky_{opt_state.nit}.png")
    plt.close()


def sample_mode_update(i):
    return "linear_resample"


def draw_linear_kwargs(i):
    return dict(cg_name="SL", cg_kwargs=dict(absdelta=absdelta / 10.0, maxiter=250))

def kl_kwargs(i):
    return dict(
        minimize_kwargs=dict(
            name="M", absdelta=absdelta, cg_kwargs=dict(name="MCG"), maxiter=20
        )
    )


key, subkey = random.split(key)
pos_init = jft.Vector(jft.random_like(subkey, Pol_sky.domain))
samples, state = jft.optimize_kl(
    nll,
    pos_init,
    n_total_iterations=n_vi_iterations,
    n_samples=n_samples,
    key=key,
    draw_linear_kwargs=draw_linear_kwargs,
    nonlinearly_update_kwargs=dict(
        minimize_kwargs=dict(
            name="SN",
            xtol=delta,
            cg_kwargs=dict(name=None),
            maxiter=10,
        )
    ),
    kl_kwargs=kl_kwargs,
    sample_mode=sample_mode_update,
    odir=odir,
    resume=False,
    callback=callback
)