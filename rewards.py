import numpy as np
from rlgym_sim.utils.reward_functions import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils import common_values

class SpeedTowardBallReward(RewardFunction):
    """Rewards the agent for moving quickly toward the ball"""
    
    def reset(self, initial_state: GameState):
        pass
    
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        player_vel = player.car_data.linear_velocity
        pos_diff = state.ball.position - player.car_data.position
        dist_to_ball = np.linalg.norm(pos_diff)
        
        if dist_to_ball == 0:
            return 0.0
            
        dir_to_ball = pos_diff / dist_to_ball
        speed_toward_ball = np.dot(player_vel, dir_to_ball)

        return float(max(speed_toward_ball / common_values.CAR_MAX_SPEED, 0.0))

# class FaceBallReward(RewardFunction):
#     def reset(self, initial_state: GameState):
#         pass
    
#     def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        
        
class InAirReward(RewardFunction):
    """Rewards the agent for being in the air"""
    
    def reset(self, initial_state: GameState):
        pass
    
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0.0 if player.on_ground else 1.0


class VelocityBallToGoalReward(RewardFunction):
    """Rewards the agent for hitting the ball toward the opponent's goal"""
    
    def reset(self, initial_state: GameState):
        pass
    
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # BLUE_TEAM is 0, ORANGE_TEAM is 1
        if player.team_num == common_values.BLUE_TEAM:
            goal_y = common_values.BACK_NET_Y
        else:
            goal_y = -common_values.BACK_NET_Y

        ball_vel = state.ball.linear_velocity
        pos_diff = np.array([0, goal_y, 0]) - state.ball.position
        dist = np.linalg.norm(pos_diff)
        
        if dist == 0:
            return 0.0
            
        dir_to_goal = pos_diff / dist
        vel_toward_goal = np.dot(ball_vel, dir_to_goal)
        
        return float(max(vel_toward_goal / common_values.BALL_MAX_SPEED, 0.0))