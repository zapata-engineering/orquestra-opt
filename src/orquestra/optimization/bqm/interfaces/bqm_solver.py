import dimod


class BQMSolver(dimod.Sampler):
    """
    This method is right now just a trivial wrapper around dimod.Sampler .
    The main reason for its existence is to make it easier to keep backward compatibility.
    If we ever implement some additional functionalities for this class it won't require changing the base class of the already implemented samplers.
    """

    def solve(self, bqm: dimod.BQM) -> dimod.SampleSet:
        """
        Wrapper around the `sample` method of dimod.Sampler, for making it more intuitive to use, as some people find
        the term "sampling" confusing in the context of combinatorial optimization problems.

        Note:
            This method should not be overwritten by inheriting classes as it's intended only as a utility or "thin wrapper".
            Any logic should be included in the `sample`, `sample_qubo` or `sample_ising` methods to keep compatibiltiy with the `dimod.Sampler` class.

        """
        return self.sample(bqm)
