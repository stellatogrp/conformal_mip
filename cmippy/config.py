class Configuration:
    def __init__(self):
        self.MAX_BOUND: int = 10_000
        self.WEIGHTED_AVGS =  [0.01, 1, 5]
        self.MEMLIMIT = 1
        self.CB_TIME_THRESH = 0.1  # seconds
        self.MAX_TERMINATION_BOUND = None

    def from_dict(self, config_dict):
        for key, value in config_dict.items():
            print(f"Setting {key} to {value}")
            setattr(self, key, value)

    def from_other_config(self, other_config):
        for key in vars(other_config):
            value = getattr(other_config, key)
            setattr(self, key, value)

CONFIG = Configuration()