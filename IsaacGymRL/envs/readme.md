# Training walk

When training a walk for the K1 or T1, use the task --task=K1 or --task=T1.
Additionally, the training is meant to be split into two phases. The first one you need to train for roughly 2k steps. Afterwards change the following parameter:

    use_swing_reward_scaling: true

This will reduce the reward function `feet_swing`, forcing the learning to optimize for the tracking rewards.

# Training kick

When training a kick policy it is recommended to use an already functioning walk policy. As the kick environment does not use a history policy, we provide checkpoints for the K1 and T1, which can be used. Additionally, we provide a checkpoint .pth file for the K1, which is already able to kick the ball quite good.

Additionally, the training is split into multiple parts:

    - 1) Train with default parameters
    - 2) Change multiple parameters
        - `target_sole_yaw_angle` to 1.5705
        - `ball_sole_yaw` up to 25
        - `disable_sole_yaw_direction_penalty` to `true`
        - `disable_deviation_flag` to `false`
        - `ball_pos_noise_x` and `ball_pos_noise_y` min value increased to 0.03
    - 3) Change multiple parameters
        - `ball_walk_speed` down to 1
        - `ball_sole_yaw` down to 15
        - `entropy` down to range `[-15, -14]`
        - `learning_rate` minimum to 1e-5
        - `target_sole_yaw_sigma` down to 0.1 in multiple steps
    - 3) Change multiple parameters
        - `disable_sole_yaw_direction_penalty` to `false`
        - `ball_sole_yaw` down to 5
        - `kick_direction_negative_factor` up to 3.0
        - `ball_walk_target_overshoot` down to -10
        - `feet_swing` down to 0.75
    - 4) `kick_direction_sigma_range` and `kick_direction_sigma_just_hit` down to 0.75, 0.5
    - 5) Change multiple parameters
        - `ball_sole_yaw` down to 0
        - `kick_direction_sigma_range` down to 0.25, 0.2, (if training looks good, also down to 0.15, 0.1, 0.05)
        - `desired_kl` to 0.003
        - `random_ball_distance_reset` up to 0.0005
        - `ball_walk_speed` down to 1.0
        - `entropy` down to range `[-16.6, -16.5]`
    - 6) Increase the following probabilities to 10% of the final value, and then to the final value. The final values only need to be used for a few hundred episodes:
        - `ball_kick_direction` down to 5.0
        - `random_ball_reset, reset_ball_request, add_ball_vel, add_ball_vel_in_direction`
        - `heavy_ball` to 0.00001 and then to 0.0001
        - Also reduce `ball_walk_speed` down to 0.75
        - `kick_pose_increase_value` up to 0.2 for the final few hundred episodes
        - `ball_distance_offset` up to 0.7
        - `ball_rotation_offset` up to 0.2

Additional note: it is recommend to first train to a point, at which the policy is able to kick somewhat accurate with strong as well as scaled kicks. If the training is moved to the next step too early, the base behavior for the accuracy as well as the strong kicks will not be learned that good later in the training. The later steps mostly focus on the last bit of accuracy.
In case you are training a T1 policy and notice it is only kicking with one leg, you can reduce the reward weight of `ball_kick_velocity_strong` down to 4 and `ball_kick_velocity` up to 8 during the initial training. Once the policy kicks with both legs, you can change both weights back to their original value.

# Training Standing Up

The training itself can be optimized by using two different sets of parameters for the block `algorithm` in the `.yaml` file.
First train with a more aggressive set of values to speed up training and ensure local stuck policy states can be resolved for 10k - 20k episodes and 1024 robots.
Afterwards use a checkpoint that can stand up and use the second set of parameters and more robots for further fine tuning.

You can also test arround with lower entropy values during training to get higher success rate for the stand up, judged by the metric `fall`.
The ideal metric value for the `fall` reward should be between `[-0.2, 0.0]`. Unfortunatly, the current provided version does not reach this reproducible, only values between `[-0.6, -0.3]`.

Note, that the **K1_fast.onnx** policy is our fast stand up for the K1. It was trained with **mjlab_playground** and requires its own safety checks. Also we only use it on the real K1 and not in simulation. For unknown reasons it works fine on the real robot but not in our simulation, but works fine in a stand-alone mujoco script. 

# Motor Parameters

For the K1, the real robot uses lower stiffness values for the ankle roll (65 -> 25).
For the T1, the real robot uses lower stiffness values for the hip rolls (250 -> 200).
This is due to some unknown sim-2-real gap. We recorded some simple joint movement while the robot was hanging in the air and repeated this experiment in IsaacGym. Those changes reduced the difference in joint behavior by a large margin.

Exception is the fast stand up for the K1. Its parameters can be looked up in the motorParameters_<robot type>.cfg file. In general we recommend using the parameters from this config file for the real robots and comparing with the c++ code example, which parameter set should be used.

