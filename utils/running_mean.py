import torch
from torch import Tensor
from torchmetrics.metric import Metric
from typing import Union, Callable, Literal, Any, Optional

Metric.dtype = torch.float32
Metric.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class BaseAggregator(Metric):

    is_differentiable = None
    higher_is_better = None
    full_state_update: bool = False

    def __init__(
        self,
        fn: Union[Callable, str],
        default_value: Union[Tensor, list],
        nan_strategy: Union[Literal["error", "warn", "ignore", "disable"], float] = "error",
        state_name: str = "value",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        allowed_nan_strategy = ("error", "warn", "ignore", "disable")
        if nan_strategy not in allowed_nan_strategy and not isinstance(nan_strategy, float):
            raise ValueError(
                f"Arg `nan_strategy` should either be a float or one of {allowed_nan_strategy} but got {nan_strategy}."
            )

        self.nan_strategy = nan_strategy
        self.add_state(state_name, default=default_value, dist_reduce_fx=fn)
        self.state_name = state_name

    def _cast_and_nan_check_input(
        self, x: Union[float, Tensor], weight: Optional[Union[float, Tensor]] = None
    ) -> tuple[Tensor, Tensor]:
        """Convert input ``x`` to a tensor and check for Nans."""
        if not isinstance(x, Tensor):
            x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        if weight is not None and not isinstance(weight, Tensor):
            weight = torch.as_tensor(weight, dtype=self.dtype, device=self.device)

        if self.nan_strategy != "disable":
            nans = torch.isnan(x)
            if weight is not None:
                nans_weight = torch.isnan(weight)
            else:
                nans_weight = torch.zeros_like(nans).bool()
                weight = torch.ones_like(x)
            if nans.any() or nans_weight.any():
                if self.nan_strategy == "error":
                    raise RuntimeError("Encountered `nan` values in tensor")
                if self.nan_strategy in ("ignore", "warn"):
                    if self.nan_strategy == "warn":
                        rank_zero_warn("Encountered `nan` values in tensor. Will be removed.", UserWarning)
                    x = x[~(nans | nans_weight)]
                    weight = weight[~(nans | nans_weight)]
                else:
                    if not isinstance(self.nan_strategy, float):
                        raise ValueError(f"`nan_strategy` shall be float but you pass {self.nan_strategy}")
                    x[nans | nans_weight] = self.nan_strategy
                    weight[nans | nans_weight] = 1
        else:
            weight = torch.ones_like(x)
        return x.to(self.dtype), weight.to(self.dtype)

    def update(self, value: Union[float, Tensor]) -> None:
        """Overwrite in child class."""

    def compute(self) -> Tensor:
        """Compute the aggregated value."""
        return getattr(self, self.state_name)

class MeanMetric(BaseAggregator):

    mean_value: Tensor
    weight: Tensor

    def __init__(
        self,
        nan_strategy: Union[Literal["error", "warn", "ignore", "disable"], float] = "warn",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            "sum",
            torch.tensor(0.0, dtype=torch.get_default_dtype()),
            nan_strategy,
            state_name="mean_value",
            **kwargs,
        )
        self.add_state("weight", default=torch.tensor(0.0, dtype=torch.get_default_dtype()), dist_reduce_fx="sum")

    def update(self, value: Union[float, Tensor], weight: Union[float, Tensor, None] = None) -> None:
        """Update state with data.

        Args:
            value: Either a float or tensor containing data. Additional tensor
                dimensions will be flattened
            weight: Either a float or tensor containing weights for calculating
                the average. Shape of weight should be able to broadcast with
                the shape of `value`. Default to None corresponding to simple
                harmonic average.

        """
        # broadcast weight to value shape
        if not isinstance(value, Tensor):
            value = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        if weight is None:
            weight = torch.ones_like(value)
        elif not isinstance(weight, Tensor):
            weight = torch.as_tensor(weight, dtype=self.dtype, device=self.device)
        weight = torch.broadcast_to(weight, value.shape)
        value, weight = self._cast_and_nan_check_input(value, weight)

        if value.numel() == 0:
            return
        self.mean_value += (value * weight).sum()
        self.weight += weight.sum()

    def compute(self) -> Tensor:
        """Compute the aggregated value."""
        return self.mean_value / self.weight
    
class Running(Metric):

    def __init__(self, base_metric: Metric, window: int = 5) -> None:
        super().__init__()
        if not isinstance(base_metric, Metric):
            raise ValueError(
                f"Expected argument `metric` to be an instance of `torchmetrics.Metric` but got {base_metric}"
            )
        if not (isinstance(window, int) and window > 0):
            raise ValueError(f"Expected argument `window` to be a positive integer but got {window}")
        self.base_metric = base_metric
        self.window = window

        if base_metric.full_state_update is not False:
            raise ValueError(
                f"Expected attribute `full_state_update` set to `False` but got {base_metric.full_state_update}"
            )
        self._num_vals_seen = 0

        for key in base_metric._defaults:
            for i in range(window):
                self.add_state(
                    name=key + f"_{i}", default=base_metric._defaults[key], dist_reduce_fx=base_metric._reductions[key]
                )

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update the underlying metric and save state afterwards."""
        val = self._num_vals_seen % self.window
        self.base_metric.update(*args, **kwargs)
        for key in self.base_metric._defaults:
            setattr(self, key + f"_{val}", getattr(self.base_metric, key))
        self.base_metric.reset()
        self._num_vals_seen += 1

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward input to the underlying metric and save state afterwards."""
        val = self._num_vals_seen % self.window
        res = self.base_metric.forward(*args, **kwargs)
        for key in self.base_metric._defaults:
            setattr(self, key + f"_{val}", getattr(self.base_metric, key))
        self.base_metric.reset()
        self._num_vals_seen += 1
        self._computed = None
        return res

    def compute(self) -> Any:
        """Compute the metric over the running window."""
        for i in range(self.window):
            self.base_metric._reduce_states({key: getattr(self, key + f"_{i}") for key in self.base_metric._defaults})
        self.base_metric._update_count = self._num_vals_seen
        val = self.base_metric.compute()
        self.base_metric.reset()
        return val

    def reset(self) -> None:
        """Reset metric."""
        super().reset()
        self._num_vals_seen = 0

class RunningMean(Running):

    def __init__(
        self,
        window: int = 5,
        nan_strategy: Union[Literal["error", "warn", "ignore", "disable"], float] = "warn",
        **kwargs: Any,
    ) -> None:
        super().__init__(base_metric=MeanMetric(nan_strategy=nan_strategy, **kwargs), window=window)
