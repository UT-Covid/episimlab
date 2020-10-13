import xsimlab as xs
import xarray as xr
import attr
from .. import seir, setup, apply_counts_delta

@attr.s
class SingleCitySEIR(object):
    model = xs.Model(dict(
        init_phi=setup.InitPhi,
        init_midx_mapping=setup.InitMidxMapping,
        foi=seir.BruteForceFOI,
        seir=seir.BruteForceSEIR,
        apply_counts_delta=apply_counts_delta.ApplyCountsDelta
    ))

    input_ds = attr.ib(default=xs.create_setup(
        model=model,
        clocks=dict(stap=range(100)),
        # master_clock=None,
        input_vars={

        },
        output_vars=None
    ))

    parallel = attr.ib(default=False)
    # TODO: validator
    scheduler = attr.ib(default='threads')
    # store = attr.ib(default=zarr.TempStore())
    store = attr.ib(default=None)

    def run(self):
        self.output_ds = self.input_ds.xsimlab.run(
            model=self.model,
            parallel=self.parallel,
            scheduler=self.scheduler,
            # store=zarr.TempStore()
        )
        return self.output_ds
