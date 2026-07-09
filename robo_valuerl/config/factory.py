"""
Factory pattern config file - automatically creates the agent and algorithm based on config type
"""
from typing import Any
from algorithm.base_algorithm import BaseAlgorithm


class FilteredBCPIAlgorithmFactory:
    """FilteredBCPI algorithm factory"""

    def create_algorithm(self, agent: Any, config: Any, is_ddp: bool, local_rank: int) -> BaseAlgorithm:
        from algorithm.filtered_bc_pi import FilteredBCPIAlgorithm
        if hasattr(config, 'algorithm'):
            algorithm_config = config.algorithm
        else:
            algorithm_config = config

        return FilteredBCPIAlgorithm(agent, algorithm_config, is_ddp, local_rank)


def create_agent_from_config(config: Any) -> Any:
    """Convenience function to create an agent from config - kept for backward compatibility"""
    # Not actually used in the current code, but the interface is kept in case it's needed in the future
    raise NotImplementedError("create_agent_from_config is not currently implemented; create the corresponding agent directly")

def create_algorithm_from_config(agent: Any, config: Any, is_ddp: bool, local_rank: int) -> BaseAlgorithm:
    """Convenience function to create an algorithm from config"""
    factory = FilteredBCPIAlgorithmFactory()
    return factory.create_algorithm(agent, config, is_ddp, local_rank)
