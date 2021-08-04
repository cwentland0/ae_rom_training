class TimeStepper:
    """Base class for time stepper models.

    There is no general form for time steppers, so methods are mostly left up
    to child class implementations.
    """

    def __init__(self, net_idx, mllib):

        self.net_idx = net_idx
        self.mllib = mllib
