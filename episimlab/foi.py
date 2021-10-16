import xarray as xr
import xsimlab as xs
from .utils import any_negative, suffixed_dims


@xs.process
class BaseFOI:
    """Base class for calculating force of infection (FOI)."""
    TAGS = ('FOI',)
    PHI_DIMS = ('age0', 'age1', 'risk0', 'risk1', 'vertex0', 'vertex1',)
    I_COMPT_LABELS = ('I')
    S_COMPT_LABELS = ('S')

    phi = xs.variable(dims=PHI_DIMS, global_name='phi', intent='in')
    state = xs.global_ref('state', intent='in')
    beta = xs.variable(global_name='beta', intent='in')
    _coords = xs.group_dict('coords')

    @property
    def phi_dims(self):
        return self.PHI_DIMS
    
    @property
    def foi_dims(self) -> tuple:
        """We assume that we want to include all phi dimensions in the FOI
        calculation.
        """
        return tuple(set(dim.rstrip('01') for dim in self.phi_dims))
    
    @property
    def foi_coords(self) -> dict:
        return {dim: self.coords[dim] for dim in self.foi_dims}
        
    @property
    def foi(self) -> xr.DataArray:
        zero_suffix = suffixed_dims(self.state[dict(compt=0)], '0')
        one_suffix = suffixed_dims(self.state[dict(compt=0)], '1')
        S = self.S.rename(zero_suffix)
        I = self.I.rename(one_suffix)
        N = self.state.sum('compt').rename(one_suffix)
        foi = ((self.beta * self.phi * S * I / N)
               # sum over coords that are not compt
               .sum(one_suffix.values())
               # like .rename({'age0': 'age', 'risk0': 'risk'})
               .rename({v: k for k, v in zero_suffix.items()}))
        
        # DEBUG
        assert not any_negative(foi, raise_err=True)
        
        return foi
    
    @property
    def I(self):
        return self.state.loc[dict(compt=self.I_COMPT_LABELS)]

    @property
    def S(self):
        return self.state.loc[dict(compt=self.S_COMPT_LABELS)]



@xs.process
class VaccineFOI(BaseFOI):
    """Base class for calculating force of infection (FOI)."""
    TAGS = ('FOI',)
    beta_reduction = xs.variable(global_name='beta_reduction', intent='in')

    @property
    def foi(self) -> xr.DataArray:
        zero_suffix = suffixed_dims(self.state[dict(compt=0)], '0')
        one_suffix = suffixed_dims(self.state[dict(compt=0)], '1')
        S = self.S.rename(zero_suffix)
        I = self.I.rename(one_suffix)
        N = self.state.sum('compt').rename(one_suffix)
        foi = ((self.beta * self.beta_reduction * self.phi * S * I / N)
               # sum over coords that are not compt
               .sum(one_suffix.values())
               # like .rename({'age0': 'age', 'risk0': 'risk'})
               .rename({v: k for k, v in zero_suffix.items()}))

        # DEBUG
        assert not any_negative(foi, raise_err=True)

        return foi


@xs.process
class BruteForceFOI(BaseFOI):
    """Calculate force of infection (FOI) using naive for looping.
    Similar to BruteForceFOI process in Episimlab v1
    """
    TAGS = ('model::SIR', 'FOI', 'brute_force')
    PHI_DIMS = ('age0', 'age1', 'risk0', 'risk1', 'vertex0', 'vertex1',)

    def run_step(self):
        self.rate_S2I = self.foi
    
    @property
    def foi(self) -> xr.DataArray:
        """Brute force FOI, like BruteForceFOI in Episimlab v1.0"""
        foi = xr.DataArray(data=0., dims=self.foi_dims, coords=self.foi_coords)
        for a0, r0, v0, a1, r1, v1 in product(*[self.age_coords, self.risk_coords, self.vertex_coords, ] * 2):
            i0, i1 = dict(vertex=v0, age=a0, risk=r0), dict(vertex=v1, age=a1, risk=r1)
            phi = self.phi.loc[dict(age0=a0, age1=a1, risk0=r0, risk1=r1, vertex0=v0, vertex1=v1)].values
            S = self.state.loc[dict(compt='S')].loc[i0].values
            I = self.state.loc[dict(compt='I')].loc[i1].values
            N = self.state.loc[i1].sum('compt').values
            foi.loc[i0] += phi * self.beta * S * I / N
        return foi
