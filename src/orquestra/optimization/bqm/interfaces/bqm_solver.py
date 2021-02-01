import dimod
from openfermion import IsingOperator
from dimod.meta import samplemixinmethod


class BQMSolver(dimod.Sampler):
    """
    This method is right now just a trivial wrapper around dimod.Sampler .
    The main reason for its existence is to make it easier to keep backward compatibility if we ever implement,
    some additional functionalities for this class it won't require changing the base class of the already implemented samplers.
    """

    pass
