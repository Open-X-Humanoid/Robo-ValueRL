import dataclasses
from algorithm.base_algorithm import BaseAlgorithm
from algorithm.bc import BCAlgorithm
from algorithm.filtered_bc_pi import FilteredBCPIAlgorithm


@dataclasses.dataclass(frozen=True)
class BCAlgorithmConfig:
    encoder_lr: float = 1e-3
    policy_lr: float = 1e-3
    scheduler_step_size: int = 10000
    scheduler_gamma: float = 0.1

    algorithm_type: BaseAlgorithm = BCAlgorithm


@dataclasses.dataclass(frozen=True)
class FilteredBCPIAlgorithmConfig:
    # --- Base config ---
    algorithm_type: BaseAlgorithm = FilteredBCPIAlgorithm

    # --- Scheduler parameters (LR Schedule) ---
    lr: float = 2e-4              # corresponds to peak_lr
    warmup_steps: int = 500      # warmup steps
    decay_steps: int = 50000     # total decay steps (usually set to total training steps)
    decay_lr: float = 2e-6        # final learning rate after decay ends (end_lr)

    # --- Optimizer parameters (AdamW) ---
    optimizer_b1: float = 0.9     # Adam betas[0]
    optimizer_b2: float = 0.95    # Adam betas[1]
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.1

    save_interval_steps: int = 2000


@dataclasses.dataclass(frozen=True)
class CalQLAlgorithmConfig:
    """
    Cal-QL algorithm config class
    Note: the corresponding CalQL algorithm implementation does not exist yet; this config is kept for future implementation
    """

    # Cal-QL-specific parameters
    cql_alpha: float = 0.1          # CQL regularization coefficient
    cql_temp: float = 1.0           # CQL temperature parameter
    cql_n_actions: int = 10        # number of actions sampled by CQL
    discount: float = 0.95          # discount factor
    tau: float = 0.005              # soft update parameter

    # Learning rate parameters
    actor_lr: float = 1e-4          # actor learning rate
    critic_lr: float = 3e-4         # critic learning rate

    # Scheduler parameters
    scheduler_step_size: int = 1000  # learning rate scheduler step size
    scheduler_gamma: float = 0.5     # learning rate scheduler decay factor

    # Target network update parameters
    target_update_freq: int = 2      # target network update frequency

    # Gradient clipping parameters
    max_grad_norm: float = 1.0      # maximum gradient norm

    # Training parameters
    batch_size: int = 32             # batch size
    num_epochs: int = 100           # number of training epochs

    # Save parameters
    save_interval: int = 100        # model save interval
    save_path: str = "./checkpoints/calql_model"  # model save path
    pretrained_ckpt_path: str = None  # pretrained model path

    # Note: algorithm_type is set to None because the CalQL implementation does not exist yet
    algorithm_type: BaseAlgorithm = None

    def __post_init__(self):
        """
        Post-init validation
        """
        # Validate parameter ranges
        assert self.cql_alpha > 0, "cql_alpha must be positive"
        assert self.cql_temp > 0, "cql_temp must be positive"
        assert self.cql_n_actions > 0, "cql_n_actions must be positive"
        assert 0 < self.discount <= 1, "discount must be in (0, 1]"
        assert 0 < self.tau <= 1, "tau must be in (0, 1]"
        assert self.actor_lr > 0, "actor_lr must be positive"
        assert self.critic_lr > 0, "critic_lr must be positive"
        assert self.scheduler_step_size > 0, "scheduler_step_size must be positive"
        assert 0 < self.scheduler_gamma <= 1, "scheduler_gamma must be in (0, 1]"
        assert self.target_update_freq > 0, "target_update_freq must be positive"
        assert self.max_grad_norm > 0, "max_grad_norm must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.num_epochs > 0, "num_epochs must be positive"
        assert self.save_interval > 0, "save_interval must be positive"


@dataclasses.dataclass(frozen=True)
class IQLAlgorithmConfig:
    """
    IQL (Implicit Q-Learning) algorithm config class
    Note: the corresponding IQL algorithm implementation does not exist yet; this config is kept for future implementation
    """

    # --- IQL-specific parameters ---
    expectile: float = 0.7          # quantile for the V-function expectile regression
    temp: float = 3.0               # temperature parameter for actor AWR (advantage weighting)
    discount: float = 0.95          # discount factor
    tau: float = 0.005              # soft update parameter

    # --- Learning rate parameters ---
    actor_lr: float = 1e-4          # actor learning rate
    critic_lr: float = 3e-4         # critic (Q-net) learning rate
    value_lr: float = 3e-4          # value (V-net) learning rate

    # --- Scheduler parameters ---
    scheduler_step_size: int = 1000  # learning rate scheduler step size
    scheduler_gamma: float = 0.9     # learning rate scheduler decay factor

    # --- Target network update parameters ---
    target_update_freq: int = 2      # target network update frequency

    # --- Gradient clipping parameters ---
    max_grad_norm: float = 1.0      # maximum gradient norm

    # --- Training parameters ---
    batch_size: int = 32             # batch size
    num_epochs: int = 100           # number of training epochs

    # --- Save parameters ---
    save_interval: int = 100        # model save interval
    save_path: str = "./checkpoints/iql_model"  # model save path

    # Note: algorithm_type is set to None because the IQL implementation does not exist yet
    algorithm_type: BaseAlgorithm = None

    def __post_init__(self):
        """
        Post-init validation
        """
        # Validate parameter ranges
        assert 0 < self.expectile < 1, "expectile must be in (0, 1)"
        assert self.temp > 0, "temp must be positive"
        assert 0 < self.discount <= 1, "discount must be in (0, 1]"
        assert 0 < self.tau <= 1, "tau must be in (0, 1]"

        assert self.actor_lr > 0, "actor_lr must be positive"
        assert self.critic_lr > 0, "critic_lr must be positive"
        assert self.value_lr > 0, "value_lr must be positive"

        assert self.scheduler_step_size > 0, "scheduler_step_size must be positive"
        assert 0 < self.scheduler_gamma <= 1, "scheduler_gamma must be in (0, 1]"
        assert self.target_update_freq > 0, "target_update_freq must be positive"
        assert self.max_grad_norm > 0, "max_grad_norm must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.num_epochs > 0, "num_epochs must be positive"
        assert self.save_interval > 0, "save_interval must be positive"


@dataclasses.dataclass(frozen=True)
class TDAlgorithmConfig:
    """
    TD algorithm config class
    Note: the corresponding TD algorithm implementation does not exist yet; this config is kept for future implementation
    """

    cql_alpha: float = 0.1          # CQL regularization coefficient
    cql_temp: float = 1.0           # CQL temperature parameter
    cql_n_actions: int = 10        # number of actions sampled by CQL
    discount: float = 0.95          # discount factor
    tau: float = 0.005              # soft update parameter

    # Learning rate parameters
    actor_lr: float = 1e-4          # actor learning rate
    critic_lr: float = 3e-4         # critic learning rate

    # Scheduler parameters
    scheduler_step_size: int = 1000  # learning rate scheduler step size
    scheduler_gamma: float = 0.9     # learning rate scheduler decay factor

    # Target network update parameters
    target_update_freq: int = 2      # target network update frequency

    # Gradient clipping parameters
    max_grad_norm: float = 1.0      # maximum gradient norm

    # Training parameters
    batch_size: int = 32             # batch size
    num_epochs: int = 100           # number of training epochs

    # Save parameters
    save_interval: int = 100        # model save interval
    save_path: str = "./checkpoints/calql_model"  # model save path

    # Note: algorithm_type is set to None because the TD implementation does not exist yet
    algorithm_type: BaseAlgorithm = None

    def __post_init__(self):
        """
        Post-init validation
        """
        # Validate parameter ranges
        assert self.cql_alpha > 0, "cql_alpha must be positive"
        assert self.cql_temp > 0, "cql_temp must be positive"
        assert self.cql_n_actions > 0, "cql_n_actions must be positive"
        assert 0 < self.discount <= 1, "discount must be in (0, 1]"
        assert 0 < self.tau <= 1, "tau must be in (0, 1]"
        assert self.actor_lr > 0, "actor_lr must be positive"
        assert self.critic_lr > 0, "critic_lr must be positive"
        assert self.scheduler_step_size > 0, "scheduler_step_size must be positive"
        assert 0 < self.scheduler_gamma <= 1, "scheduler_gamma must be in (0, 1]"
        assert self.target_update_freq > 0, "target_update_freq must be positive"
        assert self.max_grad_norm > 0, "max_grad_norm must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.num_epochs > 0, "num_epochs must be positive"
        assert self.save_interval > 0, "save_interval must be positive"


@dataclasses.dataclass(frozen=True)
class ConsistencyAlgorithmConfig:
    """
    Consistency algorithm config class
    Note: the corresponding Consistency algorithm implementation does not exist yet; this config is kept for future implementation
    """

    encoder_lr: float = 3e-4
    policy_lr: float = 3e-4
    steps_per_epoch: int = 100
    num_epochs: int = 200

    # Note: algorithm_type is set to None because the Consistency implementation does not exist yet
    algorithm_type: BaseAlgorithm = None
