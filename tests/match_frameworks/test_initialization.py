import pytest
import torch
import jax
import jax.numpy as jnp
import numpy as np

from plstm.config.initialization import (
    ConstantInitConfig,
    OnesInitConfig,
    ZerosInitConfig,
    LinspaceInitConfig,
    NormalInitConfig,
    TruncatedNormalInitConfig,
    WangInitConfig,
    SmallInitConfig,
)
from plstm.nnx.initialization import (
    ZerosInit as NNXZerosInit,
    OnesInit as NNXOnesInit,
    ConstantInit as NNXConstantInit,
    LinspaceInit as NNXLinspaceInit,
    NormalInit as NNXNormalInit,
    TruncatedNormalInit as NNXTruncatedNormalInit,
    WangInit as NNXWangInit,
    SmallInit as NNXSmallInit,
)
from plstm.torch.initialization import (
    ZerosInit as TorchZerosInit,
    OnesInit as TorchOnesInit,
    ConstantInit as TorchConstantInit,
    LinspaceInit as TorchLinspaceInit,
    NormalInit as TorchNormalInit,
    TruncatedNormalInit as TorchTruncatedNormalInit,
    WangInit as TorchWangInit,
    SmallInit as TorchSmallInit,
)
from plstm.linen.initialization import (
    ZerosInit as LinenZerosInit,
    OnesInit as LinenOnesInit,
    ConstantInit as LinenConstantInit,
    LinspaceInit as LinenLinspaceInit,
    NormalInit as LinenNormalInit,
    TruncatedNormalInit as LinenTruncatedNormalInit,
    WangInit as LinenWangInit,
    SmallInit as LinenSmallInit,
)
from plstm.test.util import request_pytest_filepath
from plstm.test.numerics import assert_allclose_with_plot
import itertools


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


class TestInitialization:
    @pytest.mark.parametrize("seed", [0, 42, 123])  # Test with different random seeds
    @pytest.mark.parametrize(
        "shape",
        [
            (32,),  # 1D tensor
            (16, 32),  # 2D tensor
            (8, 16, 32),  # 3D tensor
            (4, 8, 16, 32),  # 4D tensor
        ],
        ids=[
            "1d",
            "2d",
            "3d",
            "4d",
        ],
    )
    def test_zeros_init(self, shape, seed, rng, request):
        """Test ZerosInit and verify NNX, Linen, and PyTorch implementations
        match."""
        test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        rng = jax.random.PRNGKey(seed)
        rng_nnx, rng_linen = jax.random.split(rng)

        # Create config
        config = ZerosInitConfig()

        # Create initializers
        nnx_init = NNXZerosInit(config)
        torch_init = TorchZerosInit(config)
        linen_init = LinenZerosInit(config)

        # Initialize tensors
        nnx_tensor = nnx_init(rng_nnx, shape, jnp.float32)
        linen_tensor = linen_init(rng_linen, shape, jnp.float32)
        torch_tensor = torch.empty(shape, dtype=torch.float32)
        torch_init(torch_tensor)

        # Compare outputs
        assert_allclose_with_plot(
            np.array(nnx_tensor),
            torch_tensor.cpu().detach().numpy(),
            rtol=1e-5,
            atol=1e-5,
            base_path=f"{test_name}_{next(counter)}",
        )

        assert_allclose_with_plot(
            np.array(nnx_tensor),
            np.array(linen_tensor),
            rtol=1e-5,
            atol=1e-5,
            base_path=f"{test_name}_{next(counter)}",
        )

        assert_allclose_with_plot(
            np.array(linen_tensor),
            torch_tensor.cpu().detach().numpy(),
            rtol=1e-5,
            atol=1e-5,
            base_path=f"{test_name}_{next(counter)}",
        )

    @pytest.mark.parametrize("seed", [0, 42, 123])  # Test with different random seeds
    @pytest.mark.parametrize(
        "shape",
        [
            (32,),  # 1D tensor
            (16, 32),  # 2D tensor
            (8, 16, 32),  # 3D tensor
            (4, 8, 16, 32),  # 4D tensor
        ],
        ids=[
            "1d",
            "2d",
            "3d",
            "4d",
        ],
    )
    def test_ones_init(self, shape, seed, rng, request):
        """Test OnesInit and verify NNX, Linen, and PyTorch implementations
        match."""
        test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        rng = jax.random.PRNGKey(seed)
        rng_nnx, rng_linen = jax.random.split(rng)

        # Create config
        config = OnesInitConfig()

        # Create initializers
        nnx_init = NNXOnesInit(config)
        torch_init = TorchOnesInit(config)
        linen_init = LinenOnesInit(config)

        # Initialize tensors
        nnx_tensor = nnx_init(rng_nnx, shape, jnp.float32)
        linen_tensor = linen_init(rng_linen, shape, jnp.float32)
        torch_tensor = torch.empty(shape, dtype=torch.float32)
        torch_init(torch_tensor)

        # Compare outputs
        assert_allclose_with_plot(
            np.array(nnx_tensor),
            torch_tensor.cpu().detach().numpy(),
            rtol=1e-5,
            atol=1e-5,
            base_path=f"{test_name}_{next(counter)}",
        )

        assert_allclose_with_plot(
            np.array(nnx_tensor),
            np.array(linen_tensor),
            rtol=1e-5,
            atol=1e-5,
            base_path=f"{test_name}_{next(counter)}",
        )

        assert_allclose_with_plot(
            np.array(linen_tensor),
            torch_tensor.cpu().detach().numpy(),
            rtol=1e-5,
            atol=1e-5,
            base_path=f"{test_name}_{next(counter)}",
        )

    @pytest.mark.parametrize("seed", [0, 42, 123])  # Test with different random seeds
    @pytest.mark.parametrize(
        "shape,value",
        [
            ((32,), 0.5),  # 1D tensor with value 0.5
            ((16, 32), -1.0),  # 2D tensor with value -1.0
            ((8, 16, 32), 2.0),  # 3D tensor with value 2.0
            ((4, 8, 16, 32), 3.14),  # 4D tensor with value 3.14
        ],
        ids=[
            "1d-0.5",
            "2d-neg1",
            "3d-2.0",
            "4d-pi",
        ],
    )
    def test_constant_init(self, shape, value, seed, rng, request):
        """Test ConstantInit and verify NNX, Linen, and PyTorch implementations
        match."""
        test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        rng = jax.random.PRNGKey(seed)
        rng_nnx, rng_linen = jax.random.split(rng)

        # Create config
        config = ConstantInitConfig(value=value)

        # Create initializers
        nnx_init = NNXConstantInit(config)
        torch_init = TorchConstantInit(config)
        linen_init = LinenConstantInit(config)

        # Initialize tensors
        nnx_tensor = nnx_init(rng_nnx, shape, jnp.float32)
        linen_tensor = linen_init(rng_linen, shape, jnp.float32)
        torch_tensor = torch.empty(shape, dtype=torch.float32)
        torch_init(torch_tensor)

        # Compare outputs
        assert_allclose_with_plot(
            np.array(nnx_tensor),
            torch_tensor.cpu().detach().numpy(),
            rtol=1e-5,
            atol=1e-5,
            base_path=f"{test_name}_{next(counter)}",
        )

        assert_allclose_with_plot(
            np.array(nnx_tensor),
            np.array(linen_tensor),
            rtol=1e-5,
            atol=1e-5,
            base_path=f"{test_name}_{next(counter)}",
        )

        assert_allclose_with_plot(
            np.array(linen_tensor),
            torch_tensor.cpu().detach().numpy(),
            rtol=1e-5,
            atol=1e-5,
            base_path=f"{test_name}_{next(counter)}",
        )

    @pytest.mark.parametrize("seed", [0, 42, 123])  # Test with different random seeds
    @pytest.mark.parametrize(
        "shape,low,high,axis",
        [
            ((32,), 0.0, 1.0, 0),  # 1D tensor, axis 0
            ((16, 32), -1.0, 1.0, 1),  # 2D tensor, axis 1
            ((8, 16, 32), 0.0, 2.0, 2),  # 3D tensor, axis 2
            ((4, 8, 16, 32), -2.0, 2.0, 3),  # 4D tensor, axis 3
            ((4, 8, 16, 32), 0.0, 1.0, -1),  # 4D tensor, axis -1 (last axis)
        ],
        ids=[
            "1d-axis0",
            "2d-axis1",
            "3d-axis2",
            "4d-axis3",
            "4d-axis-1",
        ],
    )
    def test_linspace_init(self, shape, low, high, axis, seed, rng, request):
        """Test LinspaceInit and verify NNX, Linen, and PyTorch implementations
        match."""
        test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        rng = jax.random.PRNGKey(seed)
        rng_nnx, rng_linen = jax.random.split(rng)

        # Create config
        config = LinspaceInitConfig(low=low, high=high, axis=axis)

        # Create initializers
        nnx_init = NNXLinspaceInit(config)
        torch_init = TorchLinspaceInit(config)
        linen_init = LinenLinspaceInit(config)

        # Initialize tensors
        nnx_tensor = nnx_init(rng_nnx, shape, jnp.float32)
        linen_tensor = linen_init(rng_linen, shape, jnp.float32)
        torch_tensor = torch.empty(shape, dtype=torch.float32)
        torch_init(torch_tensor)

        # Compare outputs
        assert_allclose_with_plot(
            np.array(nnx_tensor),
            torch_tensor.cpu().detach().numpy(),
            rtol=1e-5,
            atol=1e-5,
            base_path=f"{test_name}_{next(counter)}",
        )

        assert_allclose_with_plot(
            np.array(nnx_tensor),
            np.array(linen_tensor),
            rtol=1e-5,
            atol=1e-5,
            base_path=f"{test_name}_{next(counter)}",
        )

        assert_allclose_with_plot(
            np.array(linen_tensor),
            torch_tensor.cpu().detach().numpy(),
            rtol=1e-5,
            atol=1e-5,
            base_path=f"{test_name}_{next(counter)}",
        )

    @pytest.mark.parametrize("seed", [0, 42, 123])  # Test with different random seeds
    @pytest.mark.parametrize(
        "shape,mean,stddev",
        [
            ((32,), 0.0, 1.0),  # 1D tensor, standard normal
            ((16, 32), 2.0, 0.5),  # 2D tensor, mean 2.0, stddev 0.5
            ((8, 16, 32), -1.0, 2.0),  # 3D tensor, mean -1.0, stddev 2.0
            ((4, 8, 16, 32), 0.0, 0.1),  # 4D tensor, mean 0.0, stddev 0.1
        ],
        ids=[
            "1d-std-normal",
            "2d-mean2-std0.5",
            "3d-mean-1-std2",
            "4d-mean0-std0.1",
        ],
    )
    def test_normal_init(self, shape, mean, stddev, seed, rng, request):
        """Test NormalInit and verify NNX, Linen, and PyTorch implementations
        match."""
        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        rng = jax.random.PRNGKey(seed)
        rng_nnx, rng_linen = jax.random.split(rng)

        # Create config
        config = NormalInitConfig(mean=mean, stddev=stddev)

        # Create initializers
        nnx_init = NNXNormalInit(config)
        torch_init = TorchNormalInit(config)
        linen_init = LinenNormalInit(config)

        # Initialize tensors
        nnx_tensor = nnx_init(rng_nnx, shape, jnp.float32)
        linen_tensor = linen_init(rng_linen, shape, jnp.float32)
        torch_tensor = torch.empty(shape, dtype=torch.float32)
        torch_init(torch_tensor)

        # For random initializations, we can't directly compare the values
        # Instead, we can check statistical properties
        nnx_mean = np.mean(np.array(nnx_tensor))
        linen_mean = np.mean(np.array(linen_tensor))
        torch_mean = torch.mean(torch_tensor).item()
        nnx_std = np.std(np.array(nnx_tensor))
        linen_std = np.std(np.array(linen_tensor))
        torch_std = torch.std(torch_tensor).item()

        # Check that means and standard deviations are close
        assert np.isclose(nnx_mean, torch_mean, rtol=0.1, atol=0.6)
        assert np.isclose(nnx_std, torch_std, rtol=0.1, atol=0.3)
        assert np.isclose(nnx_mean, linen_mean, rtol=0.1, atol=0.6)
        assert np.isclose(nnx_std, linen_std, rtol=0.1, atol=0.3)
        assert np.isclose(linen_mean, torch_mean, rtol=0.1, atol=0.6)
        assert np.isclose(linen_std, torch_std, rtol=0.1, atol=0.3)

        # Check that means and standard deviations are close to the expected values
        # For small tensors, we need to account for sampling variance
        # The standard error of the mean is approximately stddev/sqrt(n)
        n_elements = np.prod(shape)
        expected_std_error = stddev / np.sqrt(n_elements)

        # Use a more generous tolerance for small tensors
        mean_rtol = min(0.5, 3 * expected_std_error / max(abs(mean), 1e-5))
        mean_atol = min(0.5, 3 * expected_std_error)

        # Standard deviation has higher variance in small samples
        std_rtol = min(0.5, 5 / np.sqrt(n_elements))
        std_atol = min(0.5, stddev * 0.5)

        print(f"Shape: {shape}, Elements: {n_elements}")
        print(f"NNX - Mean: {nnx_mean:.4f} (expected: {mean:.4f}), Std: {nnx_std:.4f} (expected: {stddev:.4f})")
        print(f"Linen - Mean: {linen_mean:.4f} (expected: {mean:.4f}), Std: {linen_std:.4f} (expected: {stddev:.4f})")
        print(f"Torch - Mean: {torch_mean:.4f} (expected: {mean:.4f}), Std: {torch_std:.4f} (expected: {stddev:.4f})")
        print(
            f"Tolerances - Mean: rtol={mean_rtol:.4f}, atol={mean_atol:.4f},"
            f" Std: rtol={std_rtol:.4f}, atol={std_atol:.4f}"
        )

        assert np.isclose(nnx_mean, mean, rtol=mean_rtol, atol=mean_atol)
        assert np.isclose(linen_mean, mean, rtol=mean_rtol, atol=mean_atol)
        assert np.isclose(torch_mean, mean, rtol=mean_rtol, atol=mean_atol)
        assert np.isclose(nnx_std, stddev, rtol=std_rtol, atol=std_atol)
        assert np.isclose(linen_std, stddev, rtol=std_rtol, atol=std_atol)
        assert np.isclose(torch_std, stddev, rtol=std_rtol, atol=std_atol)

    @pytest.mark.parametrize("seed", [0, 42, 123])  # Test with different random seeds
    @pytest.mark.parametrize(
        "shape,mean,stddev,lower,upper",
        [
            ((32,), 0.0, 1.0, -2.0, 2.0),  # 1D tensor, standard truncated normal
            ((16, 32), 2.0, 0.5, -1.0, 1.0),  # 2D tensor, mean 2.0, stddev 0.5
            ((8, 16, 32), -1.0, 2.0, -1.5, 1.5),  # 3D tensor, mean -1.0, stddev 2.0
            ((4, 8, 16, 32), 0.0, 0.1, -3.0, 3.0),  # 4D tensor, mean 0.0, stddev 0.1
        ],
        ids=[
            "1d-std-trunc",
            "2d-mean2-std0.5",
            "3d-mean-1-std2",
            "4d-mean0-std0.1",
        ],
    )
    def test_truncated_normal_init(self, shape, mean, stddev, lower, upper, seed, rng, request):
        """Test TruncatedNormalInit and verify NNX, Linen, and PyTorch
        implementations match."""
        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        rng = jax.random.PRNGKey(seed)
        rng_nnx, rng_linen = jax.random.split(rng)

        # Create config
        config = TruncatedNormalInitConfig(mean=mean, stddev=stddev, lower=lower, upper=upper)

        # Create initializers
        nnx_init = NNXTruncatedNormalInit(config)
        torch_init = TorchTruncatedNormalInit(config)
        linen_init = LinenTruncatedNormalInit(config)

        # Initialize tensors
        nnx_tensor = nnx_init(rng_nnx, shape, jnp.float32)
        linen_tensor = linen_init(rng_linen, shape, jnp.float32)
        torch_tensor = torch.empty(shape, dtype=torch.float32)
        torch_init(torch_tensor)

        # Check that all values are within the truncated range
        nnx_min = np.min(np.array(nnx_tensor))
        nnx_max = np.max(np.array(nnx_tensor))
        linen_min = np.min(np.array(linen_tensor))
        linen_max = np.max(np.array(linen_tensor))
        torch_min = torch.min(torch_tensor).item()
        torch_max = torch.max(torch_tensor).item()

        # add numerical epsilons
        assert nnx_min >= mean + lower * stddev - 1e-5
        assert nnx_max <= mean + upper * stddev + 1e-5
        assert linen_min >= mean + lower * stddev - 1e-5
        assert linen_max <= mean + upper * stddev + 1e-5
        assert torch_min >= mean + lower * stddev - 1e-5
        assert torch_max <= mean + upper * stddev + 1e-5

        # For random initializations, we can't directly compare the values
        # Instead, we can check statistical properties
        nnx_mean = np.mean(np.array(nnx_tensor))
        linen_mean = np.mean(np.array(linen_tensor))
        torch_mean = torch.mean(torch_tensor).item()

        # Check that means are close (standard deviations will be affected by truncation)
        assert np.isclose(nnx_mean, torch_mean, rtol=0.2, atol=0.4)
        assert np.isclose(nnx_mean, linen_mean, rtol=0.2, atol=0.4)
        assert np.isclose(linen_mean, torch_mean, rtol=0.2, atol=0.4)

    # no 1D case here
    @pytest.mark.parametrize("seed", [0, 42, 123])  # Test with different random seeds
    @pytest.mark.parametrize(
        "shape,num_blocks,mup_init_scale,axis",
        [
            ((16, 32), 2, 0.5, 1),  # 2D tensor
            ((8, 16, 32), 4, 2.0, 2),  # 3D tensor
            ((4, 8, 16, 32), 8, 1.5, 3),  # 4D tensor
            ((4, 8, 16, 32), 1, 1.0, -1),  # 4D tensor, axis -1
        ],
        ids=[
            "2d-2blocks",
            "3d-4blocks",
            "4d-8blocks",
            "4d-axis-1",
        ],
    )
    def test_wang_init(self, shape, num_blocks, mup_init_scale, axis, seed, rng, request):
        """Test WangInit and verify NNX, Linen, and PyTorch implementations
        match."""
        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        rng = jax.random.PRNGKey(seed)
        rng_nnx, rng_linen = jax.random.split(rng)

        # Create config
        config = WangInitConfig(num_blocks=num_blocks, mup_init_scale=mup_init_scale, axis=axis)

        # Create initializers
        nnx_init = NNXWangInit(config)
        torch_init = TorchWangInit(config)
        linen_init = LinenWangInit(config)

        # Initialize tensors
        nnx_tensor = nnx_init(rng_nnx, shape, jnp.float32)
        linen_tensor = linen_init(rng_linen, shape, jnp.float32)
        torch_tensor = torch.empty(shape, dtype=torch.float32)
        torch_init(torch_tensor)

        # For random initializations, we can't directly compare the values
        # Instead, we can check statistical properties
        nnx_mean = np.mean(np.array(nnx_tensor))
        linen_mean = np.mean(np.array(linen_tensor))
        torch_mean = torch.mean(torch_tensor).item()
        nnx_std = np.std(np.array(nnx_tensor))
        linen_std = np.std(np.array(linen_tensor))
        torch_std = torch.std(torch_tensor).item()

        # Check that means are close to zero
        assert np.isclose(nnx_mean, 0.0, rtol=0.2, atol=0.2)
        assert np.isclose(linen_mean, 0.0, rtol=0.2, atol=0.2)
        assert np.isclose(torch_mean, 0.0, rtol=0.2, atol=0.2)

        # Check that standard deviations are close
        assert np.isclose(nnx_std, torch_std, rtol=0.2, atol=0.2)
        assert np.isclose(nnx_std, linen_std, rtol=0.2, atol=0.2)
        assert np.isclose(linen_std, torch_std, rtol=0.2, atol=0.2)

    # no 1D case here
    @pytest.mark.parametrize("seed", [0, 42, 123])  # Test with different random seeds
    @pytest.mark.parametrize(
        "shape,mup_init_scale,axis",
        [
            ((16, 32), 0.5, 1),  # 2D tensor
            ((8, 16, 32), 2.0, 2),  # 3D tensor
            ((4, 8, 16, 32), 1.5, 3),  # 4D tensor
            ((4, 8, 16, 32), 1.0, -1),  # 4D tensor, axis -1
        ],
        ids=[
            "2d-scale0.5",
            "3d-scale2",
            "4d-scale1.5",
            "4d-axis-1",
        ],
    )
    def test_small_init(self, shape, mup_init_scale, axis, seed, rng, request):
        """Test SmallInit and verify NNX, Linen, and PyTorch implementations
        match."""
        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        rng = jax.random.PRNGKey(seed)
        rng_nnx, rng_linen = jax.random.split(rng)

        # Create config
        config = SmallInitConfig(mup_init_scale=mup_init_scale, axis=axis)

        # Create initializers
        nnx_init = NNXSmallInit(config)
        torch_init = TorchSmallInit(config)
        linen_init = LinenSmallInit(config)

        # Initialize tensors
        nnx_tensor = nnx_init(rng_nnx, shape, jnp.float32)
        linen_tensor = linen_init(rng_linen, shape, jnp.float32)
        torch_tensor = torch.empty(shape, dtype=torch.float32)
        torch_init(torch_tensor)

        # For random initializations, we can't directly compare the values
        # Instead, we can check statistical properties
        nnx_mean = np.mean(np.array(nnx_tensor))
        linen_mean = np.mean(np.array(linen_tensor))
        torch_mean = torch.mean(torch_tensor).item()
        nnx_std = np.std(np.array(nnx_tensor))
        linen_std = np.std(np.array(linen_tensor))
        torch_std = torch.std(torch_tensor).item()

        # Check that means are close to zero
        assert np.isclose(nnx_mean, 0.0, rtol=0.2, atol=0.2)
        assert np.isclose(linen_mean, 0.0, rtol=0.2, atol=0.2)
        assert np.isclose(torch_mean, 0.0, rtol=0.2, atol=0.2)

        # Check that standard deviations are close
        assert np.isclose(nnx_std, torch_std, rtol=0.2, atol=0.2)
        assert np.isclose(nnx_std, linen_std, rtol=0.2, atol=0.2)
        assert np.isclose(linen_std, torch_std, rtol=0.2, atol=0.2)
