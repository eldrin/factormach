class MultipleOptimizer:
    """ Simple wrapper for the multiple optimizer scenario """
    def __init__(self, *op):
        """"""
        self.optimizers = op

    def zero_grad(self):
        """"""
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        """"""
        for op in self.optimizers:
            op.step()

    def state_dict(self):
        """"""
        return [op.state_dict() for op in self.optimizers]

    def load_state_dict(self, state_dict):
        """"""
        return [op.load_state_dict(state) for state in state_dict]
