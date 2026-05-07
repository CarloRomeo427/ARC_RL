from src.algos.agent_sac import SACAgent


class DroQAgent(SACAgent):
    """SAC + dropout & LayerNorm in critics + high UTD (Hiraoka et al., 2022)."""
    pass
