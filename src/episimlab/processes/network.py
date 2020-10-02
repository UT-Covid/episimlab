import xsimlab as xs

@xs.process
class Network:

    amx = xs.variable(dims='vertex0', intent="inout")
    cur_cts = xs.variable(intent="in")

    def run_step(self):
        """
        """
        pass

