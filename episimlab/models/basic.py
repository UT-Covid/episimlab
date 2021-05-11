import xsimlab as xs

from ..setup import seed, sto, epi, counts, coords, adj, phi
from ..foi import (
    brute_force as bf_foi,
    bf_cython as bf_cython_foi
)
from ..seir import (
    base as base_seir,
    brute_force as bf_seir,
    bf_cython as bf_cython_seir
)
from .. import apply_counts_delta
from ..partition.partition import Partition
from ..io.config import ReadV1Config
from ..network import cython_explicit_travel


def minimum_viable():
    return xs.Model(dict(
        # Random number generator
        rng=seed.SeedGenerator,
        sto=sto.InitStochasticFromToggle,

        # Instantiate coords and counts array
        setup_counts=counts.InitDefaultCounts,
        setup_coords=coords.InitCoordsFromConfig,

        # Instantiate params that inform epi params
        # setup_asymp_infect=epi.asymp_infect.SetupDefaultAsympInfect,
        # setup_hosp_f_ratio=epi.hosp_f_ratio.SetupDefaultHospFRatio,
        # setup_prop_trans=epi.prop_trans.SetupDefaultPropTransP,
        # setup_symp_h_ratio=epi.symp_h_ratio.SetupDefaultSympHRatio,
        # setup_symp_h_ratio_w_risk=epi.symp_h_ratio.SetupDefaultSympHRatioWithRisk,
        read_config=ReadV1Config,

        # Instantiate epidemiological parameters
        setup_beta=epi.SetupDefaultBeta,
        setup_eta=epi.SetupEtaFromAsympRate,
        setup_gamma=epi.SetupStaticGamma,
        setup_mu=epi.SetupStaticMuFromHtoD,
        setup_nu=epi.SetupStaticNu,
        setup_omega=epi.SetupStaticOmega,
        setup_pi=epi.SetupStaticPi,
        setup_rho=epi.SetupStaticRhoFromTri,
        setup_sigma=epi.SetupStaticSigmaFromExposedPara,
        setup_tau=epi.SetupTauFromAsympRate,

        # no SEIR engine, these are just placeholders
        foi=bf_foi.BaseFOI,
        seir=base_seir.BaseSEIR,

        # Apply all changes made to counts
        apply_counts_delta=apply_counts_delta.ApplyCountsDelta
    ))


def slow_seir():
    """Python FOI and SEIR with 11 compartments.
    """
    model = minimum_viable()
    return model.update_processes(dict(
        # Instantiate phi array
        setup_phi=phi.InitPhi,
        # Force of infection calculation in python
        foi=bf_foi.BruteForceFOI,
        # SEIR engine in python
        seir=bf_seir.BruteForceSEIR,
    ))


def cy_adj():
    model = minimum_viable()
    return model.update_processes(dict(
        # Initialize adjacency matrix
        setup_adj=adj.InitToyAdj,
        # Use adjacency matrix to simulate travel between vertices in cython
        travel=cython_explicit_travel.CythonExplicitTravel,
    ))


def cy_adj_slow_seir():
    model = slow_seir()
    return model.update_processes(dict(
        # Initialize adjacency matrix
        setup_adj=adj.InitToyAdj,
        # Use adjacency matrix to simulate travel between vertices in cython
        travel=cython_explicit_travel.CythonExplicitTravel,
    ))


def slow_seir_cy_foi():
    model = slow_seir()
    return model.update_processes(dict(
        foi=bf_cython_foi.BruteForceCythonFOI,
    ))


def cy_seir_cy_foi():
    model = slow_seir()
    return model.update_processes(dict(
        foi=bf_cython_foi.BruteForceCythonFOI,
        seir=bf_cython_seir.BruteForceCythonSEIR
    ))


def cy_seir_cy_foi_cy_adj():
    model = cy_seir_cy_foi()
    return model.update_processes(dict(
        # Initialize adjacency matrix
        setup_adj=adj.InitToyAdj,
        # Use adjacency matrix to simulate travel between vertices in cython
        travel=cython_explicit_travel.CythonExplicitTravel,
    ))


def partition():
    model = cy_seir_cy_foi()
    return model.update_processes(dict(
        setup_phi=Partition
    ))