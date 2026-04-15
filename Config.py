# Config.py
class Config(object):
    # ----------------------------
    # System setup (default: light load)
    # ----------------------------
    N_UE             = 20
    N_EDGE           = 2
    UE_COMP_CAP      = 2.6
    UE_TRAN_CAP      = 14
    EDGE_COMP_CAP    = 42

    # Energy consumption settings
    UE_ENERGY_STATE  = [0.25, 0.50, 0.75]
    UE_COMP_ENERGY   = 2
    UE_TRAN_ENERGY   = 2.3
    UE_IDLE_ENERGY   = 0.1
    EDGE_COMP_ENERGY = 5

    # Task Requirement
    TASK_COMP_DENS   = [0.197, 0.297, 0.397]
    TASK_MIN_SIZE    = 1
    TASK_MAX_SIZE    = 7
    N_COMPONENT      = 1
    MAX_DELAY        = 10

    # Simulation scenario
    N_EPISODE        = 1000
    N_TIME_SLOT      = 100
    DURATION         = 0.1
    TASK_ARRIVE_PROB = 0.50
    N_TIME           = N_TIME_SLOT + MAX_DELAY

    # Algorithm settings
    LEARNING_RATE    = 0.01
    REWARD_DECAY     = 0.9
    E_GREEDY         = 0.99
    N_NETWORK_UPDATE = 200
    MEMORY_SIZE      = 500

    # ----------------------------
    # RoSCo additions (needed by your run_comparison/main_rosco)
    # ----------------------------
    # Trust & sensitivity
    EN_TRUST_LEVELS = [0.95, 0.85, 0.70, 0.55, 0.40]  # will be truncated/used by N_EDGE
    TASK_SENSITIVITY_LEVELS = [0, 1, 2]              # 0=low,1=mid,2=high

    # Security penalty (soft term in QoE)
    SECURITY_PENALTY_MULTIPLIER = 30.0

    # Load balance reward/penalty
    LOAD_BALANCE_REWARD_SCALE = 8.0
    LOAD_THRESHOLD_OFFSET = 0.10
    LOAD_PENALTY_FACTOR = 1.0

    # ----------------------------
    # Scenario presets (for paper)
    # ----------------------------
    @classmethod
    def apply_scenario(cls, name: str) -> None:
        """
        name:
          - "easy": drop ~ 0 (当前你图里的情况)
          - "heavy": 强压测，让 drop > 0（论文更有信息量）
          - "paper": 更接近 50UE/5EN（如果你要对齐论文规模）
        """
        name = (name or "easy").lower()

        if name == "easy":
            # keep defaults
            pass
        elif name == "medium":
            cls.TASK_ARRIVE_PROB = 0.58
            cls.TASK_MIN_SIZE = 2
            cls.TASK_MAX_SIZE = 8
            cls.MAX_DELAY = 8
            cls.UE_TRAN_CAP = 12
            cls.EDGE_COMP_CAP = 36
        elif name == "heavy":
            # 强负载：让 drop 不再永远 0
            cls.TASK_ARRIVE_PROB = 0.65
            cls.TASK_MIN_SIZE = 3
            cls.TASK_MAX_SIZE = 10

            # 收紧deadline/降低资源，让队列更容易爆
            cls.MAX_DELAY = 6
            cls.UE_TRAN_CAP = 10
            cls.EDGE_COMP_CAP = 30

        elif name == "paper":
            # 可选：更“论文规模”一些
            cls.N_UE = 50
            cls.N_EDGE = 5
            cls.TASK_ARRIVE_PROB = 0.50
            cls.TASK_MIN_SIZE = 2
            cls.TASK_MAX_SIZE = 9
            cls.MAX_DELAY = 8
            cls.UE_TRAN_CAP = 12
            cls.EDGE_COMP_CAP = 42

        else:
            raise ValueError(f"Unknown scenario: {name}")

        # keep N_TIME consistent
        cls.N_TIME = cls.N_TIME_SLOT + cls.MAX_DELAY
