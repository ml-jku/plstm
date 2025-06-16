import pytest
import compoconf
from plstm.nnx.scalar import ScalarFunctionLayer as JaxScalarFunctionLayer
from plstm.torch.scalar import ScalarFunctionLayer as TorchScalarFunctionLayer
from compoconf import parse_config
import numpy as np
import jax.numpy as jnp
import torch
from plstm.test.util import request_pytest_filepath
import itertools
from plstm.test.numerics import assert_allclose_with_plot


@pytest.mark.parametrize(
    "shape",
    [
        (1,),
        (32,),
        (4, 16),
    ],
)
def test_scalar_function_match(shape: list[int], request):
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    functions = ["TanhFunctionLayer", "SigmoidFunctionLayer", "SoftCapFunctionLayer", "ExpExpFunctionLayer"]

    print(compoconf.compoconf.Registry._registries, compoconf.compoconf.Registry._registry_classes)
    for fun in functions:
        parsed_config = parse_config(JaxScalarFunctionLayer.cfgtype, {"class_name": fun})

        nnx_module = parsed_config.instantiate(JaxScalarFunctionLayer)
        torch_module = parsed_config.instantiate(TorchScalarFunctionLayer)

        x = np.random.randn(*shape)

        jax_out = nnx_module(jnp.array(x))
        torch_out = torch_module(torch.from_numpy(x))

        # Compare outputs
        assert_allclose_with_plot(
            np.array(jax_out),
            torch_out.cpu().detach().numpy(),
            rtol=2e-3,
            atol=2e-3,
            base_path=f"{test_name}_{next(counter)}",
        )
