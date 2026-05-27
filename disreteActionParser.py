from typing import Any
import numpy as np
from gym.spaces import Box 

from rlgym_sim.utils.action_parsers import ActionParser
from rlgym_sim.utils.gamestates import GameState

class DiscreteAction(ActionParser):
    def __init__(self):
        super().__init__()
        self._n_controller_inputs = 8
        
    def get_action_space(self) -> Box:
        return Box(low=-1, high=1, shape=(self._n_controller_inputs,))

    def reset(self, initial_state: GameState) -> None:
        pass

    def parse_actions(self, actions: Any, state: GameState) -> np.ndarray:
        # rlgym-sim passes actions as a numpy array. 
        # We convert to asarray to be safe, then copy to avoid modifying the original.
        controls = np.asarray(actions).copy()
        
        # Check if we have one agent (1D) or multiple agents (2D)
        if controls.ndim == 1:
            # Single agent case: indices 5, 6, 7 are Jump, Boost, Handbrake
            controls[:] = np.round((controls[:] + 1) / 2)
        else:
            # Multi-agent case: apply to the last 3 columns for all rows
            controls[:, :] = np.round((controls[:, :] + 1) / 2)
            
        return controls