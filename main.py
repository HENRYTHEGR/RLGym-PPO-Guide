import numpy as np
from rlgym_sim.utils.gamestates import GameState
from rlgym_ppo.util import MetricsLogger
import os

class ExampleLogger(MetricsLogger):
    def _collect_metrics(self, game_state: GameState) -> list:
        return [game_state.players[0].car_data.linear_velocity,
                game_state.players[0].car_data.rotation_mtx(),
                game_state.orange_score]

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        avg_linvel = np.zeros(3)
        for metric_array in collected_metrics:
            p0_linear_velocity = metric_array[0]
            avg_linvel += p0_linear_velocity
        avg_linvel /= len(collected_metrics)
        report = {"x_vel":avg_linvel[0],
                  "y_vel":avg_linvel[1],
                  "z_vel":avg_linvel[2],
                  "Cumulative Timesteps":cumulative_timesteps}
        wandb_run.log(report)


def build_rocketsim_env():
    import rlgym_sim
    from rlgym_sim.utils.reward_functions import CombinedReward
    from rlgym_sim.utils.reward_functions.common_rewards import EventReward, FaceBallReward
    from rewards import SpeedTowardBallReward, InAirReward, VelocityBallToGoalReward
    from rlgym_sim.utils.obs_builders import DefaultObs
    from rlgym_sim.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition
    from rlgym_sim.utils import common_values
    from rlgym_sim.utils.action_parsers import ContinuousAction
    from disreteActionParser import DiscreteAction

    spawn_opponents = True
    team_size = 1
    game_tick_rate = 120
    tick_skip = 8
    timeout_seconds = 10
    timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))

    action_parser = DiscreteAction()
    terminal_conditions = [NoTouchTimeoutCondition(timeout_ticks), GoalScoredCondition()]
    ## Format: (reward, weight)
    # rewards = (
    #     (EventReward(touch=1), 50), # Giant reward for actually hitting the ball
    #     (SpeedTowardBallReward(), 5), # Move towards the ball!
    #     (FaceBallReward(), 1), # Make sure we don't start driving backward at the ball
    #     (AirReward(), 0.15) # Make sure we don't forget how to jump
    # )

    rewards_to_combine = (
                        SpeedTowardBallReward(),
                        InAirReward(),
                        FaceBallReward(),
                        EventReward(touch=1),
                        VelocityBallToGoalReward()
                    )
    reward_weights = (5, .01, 1, 50, .1)

    reward_fn = CombinedReward(reward_functions=rewards_to_combine,
                               reward_weights=reward_weights)

    obs_builder = DefaultObs(
        pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL)

    env = rlgym_sim.make(tick_skip=tick_skip,
                         team_size=team_size,
                         spawn_opponents=spawn_opponents,
                         terminal_conditions=terminal_conditions,
                         reward_fn=reward_fn,
                         obs_builder=obs_builder,
                         action_parser=action_parser)
    
    # Add these lines right after "env = rlgym_sim.make( ..."
    import rocketsimvis_rlgym_sim_client as rsv
    type(env).render = lambda self: rsv.send_state_to_rocketsimvis(self._prev_state)

    # That's it!

    return env

if __name__ == "__main__":
    from rlgym_ppo import Learner
    metrics_logger = ExampleLogger()

    # 32 processes
    n_proc = 32

    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))
    checkpoint_base = "data/checkpoints/rlgym-ppo-run"
    latest_checkpoint_dir = None

    # 2. Check if the directory exists and has numbered folders
    if os.path.exists(checkpoint_base):
        # We only want folders that are digits (timesteps)
        checkpoints = [d for d in os.listdir(checkpoint_base) if d.isdigit()]
        if checkpoints:
            # Join the path correctly to the highest numbered folder
            latest_checkpoint_dir = os.path.join(checkpoint_base, max(checkpoints, key=int))
            print(f"Loading from: {latest_checkpoint_dir}")
        else:
            print("No numbered checkpoint folders found. Starting from scratch.")
    else:
        print("Checkpoint base directory not found. Starting from scratch.")

    learner = Learner(build_rocketsim_env,
                      n_proc=n_proc,
                      render=True,
                      render_delay=1/120,
                      policy_layer_sizes=[2048,2048,1024,1024],
                      critic_layer_sizes=[2048,2048,1024,1024],
                      ppo_ent_coef=0.01,
                      checkpoint_load_folder=latest_checkpoint_dir,
                      min_inference_size=min_inference_size,
                      metrics_logger=metrics_logger,
                      ppo_batch_size=50000,
                      ts_per_iteration=3000,
                      exp_buffer_size=1500000,
                      add_unix_timestamp=False,
                      ppo_minibatch_size=50000,
                      ppo_epochs=1,
                      standardize_returns=True,
                      standardize_obs=False,
                      save_every_ts=100_000,
                      timestep_limit=100_000_000_000,
                      log_to_wandb=False)
    learner.learn()