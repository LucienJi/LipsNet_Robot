import torch 
import torch.nn as nn 
class Lagrange(nn.Module):
    """Base class for Lagrangian-base Algorithms.

    This class implements the Lagrange multiplier update and the Lagrange loss.

    ..  note::
        Any traditional policy gradient algorithm can be converted to a Lagrangian-based algorithm
        by inheriting from this class and implementing the :meth:`_loss_pi` method.

    Examples:
        >>> from omnisafe.common.lagrange import Lagrange
        >>> def loss_pi(self, data):
        ...     # implement your own loss function here
        ...     return loss

    You can also inherit this class to implement your own Lagrangian-based algorithm, with any
    policy gradient method you like in OmniSafe.

    Examples:
        >>> from omnisafe.common.lagrange import Lagrange
        >>> class CustomAlgo:
        ...     def __init(self) -> None:
        ...         # initialize your own algorithm here
        ...         super().__init__()
        ...         # initialize the Lagrange multiplier
        ...         self.lagrange = Lagrange(**self._cfgs.lagrange_cfgs)

    Args:
        cost_limit (float): The cost limit.
        lagrangian_multiplier_init (float): The initial value of the Lagrange multiplier.
        lambda_lr (float): The learning rate of the Lagrange multiplier.
        lambda_optimizer (str): The optimizer for the Lagrange multiplier.
        lagrangian_upper_bound (float or None, optional): The upper bound of the Lagrange multiplier.
            Defaults to None.

    Attributes:
        cost_limit (float): The cost limit.
        lambda_lr (float): The learning rate of the Lagrange multiplier.
        lagrangian_upper_bound (float, optional): The upper bound of the Lagrange multiplier.
            Defaults to None.
        lagrangian_multiplier (torch.nn.Parameter): The Lagrange multiplier.
        lambda_range_projection (torch.nn.ReLU): The projection function for the Lagrange multiplier.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        cost_limit: float,
        lagrangian_multiplier_init: float,
        n_dim: int,
        lambda_lr: float,
        lambda_optimizer: str,
        lagrangian_upper_bound: float
    ) -> None:
        """Initialize an instance of :class:`Lagrange`."""
        super().__init__()
        self.cost_limit: float = cost_limit
        self.lambda_lr: float = lambda_lr
        self.lagrangian_upper_bound: float = lagrangian_upper_bound

        init_value = max(lagrangian_multiplier_init, 0.0)
        self.lagrangian_multiplier: torch.nn.Parameter = torch.nn.Parameter(
            torch.as_tensor([init_value for _ in range(n_dim)]),
            requires_grad=True,
        )
        self.lambda_range_projection: torch.nn.ReLU = torch.nn.ReLU()
        # fetch optimizer from PyTorch optimizer package
        assert hasattr(
            torch.optim,
            lambda_optimizer,
        ), f'Optimizer={lambda_optimizer} not found in torch.'
        torch_opt = getattr(torch.optim, lambda_optimizer)
        self.lambda_optimizer: torch.optim.Optimizer = torch_opt(
            [
                self.lagrangian_multiplier,
            ],
            lr=lambda_lr,
        )

    def compute_lambda_loss(self, k2) -> torch.Tensor:
        # lagrangian_multiplier.shape (n_action)
        # k2.shape (n_action)
        loss = -self.lagrangian_multiplier * (k2 - self.cost_limit)
        return loss.sum()

    def update_lagrange_multiplier(self, k2) -> None:
        r"""Update Lagrange multiplier (lambda).

        We update the Lagrange multiplier by minimizing the penalty loss, which is defined as:

        .. math::

            \lambda ^{'} = \lambda + \eta \cdot (J_C - J_C^*)

        where :math:`\lambda` is the Lagrange multiplier, :math:`\eta` is the learning rate,
        :math:`J_C` is the mean episode cost, and :math:`J_C^*` is the cost limit.

        Args:
            Jc (float): mean episode cost.
        """
        self.lambda_optimizer.zero_grad()
        lambda_loss = self.compute_lambda_loss(k2.detach())
        lambda_loss.backward()
        self.lambda_optimizer.step()
        self.lagrangian_multiplier.data.clamp_(
            0.0,
            self.lagrangian_upper_bound,
        )  # enforce: lambda in [0, inf]