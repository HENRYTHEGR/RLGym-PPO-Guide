from typing import Dict, Any

import numpy as np

from rlgym.api import ActionParser, AgentID
from rlgym.rocket_league.api import GameState

class DiscreteAction(ActionParser[AgentID, np.ndarray, np.ndarray, GameState, int]):
    """
    Simple discrete action space that maps an array of 8 values on the interval [-1, 1] into an array of valid car
    controls.
    """

    def __init__(self):
        super().__init__()
        # Rocket League expects 8 values per controller input.
        self._n_controller_inputs = 8
        
    def get_action_space(self) -> tuple:
        return float(self._n_controller_inputs), 'discrete'

    def reset(self, initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def parse_actions(self, actions: Dict[AgentID, np.ndarray], state: GameState, shared_info: Dict[str, Any]) -> Dict[AgentID, np.ndarray]:
        parsed_actions = {}
        
        # Loop over the agent action dictionary
        for agent, action in actions.items():
            # Copy the action into a new array
            car_controls = np.zeros(self._n_controller_inputs)
            car_controls[:] = action[:]
            
            # All the actions from our policy will be on the interval [-1, 1], but the last 3 values in the car controls
            # need to be either 0 or 1. We will shift and round the result such that any value below 0 becomes 0 and
            # any value above 0 becomes 1.
            car_controls[:] = np.round((car_controls[:] + 1) / 2)
            parsed_actions[agent] = car_controls

        return parsed_actions