

class TimeStepper:
    """Base class for time stepper models.
    
    There is no general form for time steppers, so methods are mostly left up
    to child class implementations.
    """

    def __init__(self, mllib):
        
        self.mllib = mllib

