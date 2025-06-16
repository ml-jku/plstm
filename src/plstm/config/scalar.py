from dataclasses import dataclass
from compoconf import ConfigInterface


@dataclass
class ScalarFunctionConfig(ConfigInterface):
    pass


@dataclass
class TanhFunctionConfig(ScalarFunctionConfig):
    pass


@dataclass
class SigmoidFunctionConfig(ScalarFunctionConfig):
    pass


@dataclass
class ExpExpFunctionConfig(ScalarFunctionConfig):
    scale: float = 0.1


@dataclass
class SoftCapFunctionConfig(ScalarFunctionConfig):
    scale: float = 10.0
