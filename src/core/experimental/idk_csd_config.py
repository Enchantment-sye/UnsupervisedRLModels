
from dataclasses import dataclass, field
from core.metra_config import MetraConfig, AlgoConfig

@dataclass
class IdkCsdAlgoConfig(AlgoConfig):
    dual_reg: int = 0
    contrastive_n_epochs: int = 5      # N: Representation learning epochs
    contrastive_m_epochs: int = 5      # M: Policy learning epochs
    contrastive_warmup_epochs: int = 5 # Warm-up epochs
    contrastive_temperature: float = 0.1 # NCE temperature
    idk_update_interval: int = 200 # Update IDK map interval
    contrastive_rollout_batch_size: int = 0
    contrastive_temporal_budget: float = 1.0
    contrastive_mix_schedule: str = "cosine"
    contrastive_exp_k: float = 5.0
    traj_pos_encoding: str = "rotary"
    traj_pos_encoding_base: float = 10000.0

@dataclass
class IdkCsdConfig(MetraConfig):
    algo: IdkCsdAlgoConfig = field(default_factory=IdkCsdAlgoConfig)
