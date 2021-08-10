import xsimlab as xs
import xarray as xr


@xs.process
class InitPhi:
    """
    """
    DIMS = ('vertex1', 'vertex2',
            'age_group1', 'age_group2',
            'risk_group1', 'risk_group2')

    phi = xs.variable(dims=DIMS, static=True, intent='out')
    phi_t = xs.variable(dims=DIMS, intent='out', global_name='phi_t')
    age_group = xs.global_ref('age_group')
    risk_group = xs.global_ref('risk_group')
    vertex = xs.global_ref('vertex')

    def initialize(self):
        # coords = ((dim, getattr(self, dim)) for dim in self.DIMS)
        self.COORDS = {k: getattr(self, k[:-1]) for k in self.DIMS}
        # TODO
        data = 1.
        self.phi = xr.DataArray(data=data, dims=self.DIMS, coords=self.COORDS)
        self.phi_t = self.phi

    @xs.runtime(args='step')
    def run_step(self, step):
        pass

