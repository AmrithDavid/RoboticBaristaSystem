import os
import sys
import numpy as np
import gym
from gym import spaces
import pybullet as p

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robotic_arm.environment import RoboticBaristaEnv

class SimpleRLEnv(gym.Env):
    """A simplified RL environment for the Robotic Barista"""
    
    def __init__(self, target_container="coffee", gui=True, curriculum_phase=0):
        super(SimpleRLEnv, self).__init__()
        
        # Create the PyBullet environment
        self.env = RoboticBaristaEnv(gui=gui)
        
        # Set target (0 for coffee, 1 for matcha)
        self.target_container = 0 if target_container == "coffee" else 1
        self.target_name = target_container
        
        # Get original target positions
        if self.target_container == 0:
            self.original_target_position = self.env.coffee_position
        else:
            self.original_target_position = self.env.matcha_position
        
        self.original_cup_position = self.env.cup_position
        
        # Curriculum learning (0: easy, 1: medium, 2: hard)
        self.curriculum_phase = curriculum_phase
        self._apply_curriculum()
        
        # Define action space - now using all 7 joints for better control
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.env.joints),),  # All controllable joints
            dtype=np.float32
        )
        
        # Define observation space - now with relative distances to targets
        # Includes joint positions, end effector position, distances to targets, and phase indicator
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.env.joints) + 3 + 2 + 1,),  # joints + ee pos + distances + phase
            dtype=np.float32
        )
        
        # Initialize counters
        self.steps = 0
        self.max_steps = 1000  # Increased max steps
        self.reached_container = False
        
        # Track best distances for shaping rewards
        self.best_container_distance = float('inf')
        self.best_cup_distance = float('inf')
        
        # Success reward threshold
        self.success_threshold = 0.30  # 30cm is close enough
        
        # Movement tracking
        self.position_history = []
        self.history_size = 20
    
    def _apply_curriculum(self):
        """Apply curriculum learning by adjusting target positions"""
        # Scale factor based on curriculum phase
        if self.curriculum_phase == 0:  # Easy
            scaling = 0.5  # Start with targets at 50% distance
        elif self.curriculum_phase == 1:  # Medium
            scaling = 0.75  # 75% distance
        else:  # Hard (original)
            scaling = 1.0
            
        # Update target positions
        self._adjust_target_positions(scaling)
    
    def _adjust_target_positions(self, scaling):
        """Adjust target positions based on scaling factor"""
        robot_base = [0, 0, self.env.table_height]
        
        # Scale target positions (keep Z height the same)
        def scale_position(original, base, scale):
            return [
                base[0] + (original[0] - base[0]) * scale,
                base[1] + (original[1] - base[1]) * scale,
                original[2]  # Keep original Z height
            ]
        
        # Apply scaling
        self.target_position = scale_position(self.original_target_position, robot_base, scaling)
        self.cup_position = scale_position(self.original_cup_position, robot_base, scaling)
        
        # Update visual positions of containers and cup
        if self.target_container == 0:
            p.resetBasePositionAndOrientation(
                self.env.coffee_container_id, 
                self.target_position, 
                p.getBasePositionAndOrientation(self.env.coffee_container_id)[1]
            )
        else:
            p.resetBasePositionAndOrientation(
                self.env.matcha_container_id, 
                self.target_position, 
                p.getBasePositionAndOrientation(self.env.matcha_container_id)[1]
            )
        
        p.resetBasePositionAndOrientation(
            self.env.cup_id, 
            self.cup_position, 
            p.getBasePositionAndOrientation(self.env.cup_id)[1]
        )
    
    def reset(self):
        """Reset the environment"""
        # Reset PyBullet environment
        self.env.reset()
        
        # Apply curriculum adjustments
        self._apply_curriculum()
        
        # Reset internal state
        self.steps = 0
        self.reached_container = False
        self.best_container_distance = float('inf')
        self.best_cup_distance = float('inf')
        self.position_history = []
        
        # Return initial observation
        return self._get_observation()
    
    def _get_observation(self):
        """Get current state observation"""
        # Get joint positions
        joint_positions = []
        for i in self.env.joints:
            joint_state = p.getJointState(self.env.robot_id, i)
            joint_positions.append(joint_state[0])
        
        # Get end effector position
        end_effector_state = p.getLinkState(self.env.robot_id, self.env.end_effector_index)
        end_effector_position = end_effector_state[0]
        
        # Track position history for movement analysis
        self.position_history.append(end_effector_position)
        if len(self.position_history) > self.history_size:
            self.position_history.pop(0)
        
        # Calculate distances to targets
        container_distance = np.sqrt(sum([(end_effector_position[i] - self.target_position[i])**2 for i in range(3)]))
        cup_distance = np.sqrt(sum([(end_effector_position[i] - self.cup_position[i])**2 for i in range(3)]))
        
        # Update best distances
        self.best_container_distance = min(self.best_container_distance, container_distance)
        self.best_cup_distance = min(self.best_cup_distance, cup_distance)
        
        # Phase indicator (0 = go to container, 1 = go to cup)
        phase = 1.0 if self.reached_container else 0.0
        
        # Combine into observation
        obs = np.array(
            joint_positions + 
            list(end_effector_position) + 
            [container_distance, cup_distance] +
            [phase]
        )
        return obs
    
    def step(self, action):
        """Execute action and return result"""
        # Apply scaled actions to joints with increased velocity scale
        for i, joint_idx in enumerate(self.env.joints):
            if i < len(action):
                # Scale action from [-1, 1] to reasonable joint velocities
                # Increased from 0.3 to 0.5 for more movement
                velocity = action[i] * 0.5
                p.setJointMotorControl2(
                    self.env.robot_id, 
                    joint_idx, 
                    p.VELOCITY_CONTROL, 
                    targetVelocity=velocity,
                    force=300  # Keep force moderate for control
                )
        
        # Step simulation multiple times for stability
        for _ in range(5):
            p.stepSimulation()
        
        # Get new observation
        obs = self._get_observation()
        
        # Extract distances from observation
        container_distance = obs[len(self.env.joints) + 3]  # Offset for joints and ee pos
        cup_distance = obs[len(self.env.joints) + 3 + 1]
        
        # Increment counter
        self.steps += 1
        
        # Calculate reward
        reward, done = self._compute_reward(container_distance, cup_distance)
        
        # Check if max steps reached
        if self.steps >= self.max_steps:
            done = True
        
        # Additional info
        info = {
            'steps': self.steps,
            'reached_container': self.reached_container,
            'container_distance': container_distance,
            'cup_distance': cup_distance,
            'curriculum_phase': self.curriculum_phase
        }
        
        return obs, reward, done, info
    
    def _compute_reward(self, container_distance, cup_distance):
        """Compute reward based on distances to targets"""
        # Base reward (small negative to encourage efficiency)
        reward = -0.01
        
        # Get end effector position
        end_effector_state = p.getLinkState(self.env.robot_id, self.env.end_effector_index)
        end_effector_position = end_effector_state[0]
        
        # Calculate distance from base (XY plane only)
        base_xy_distance = np.sqrt(end_effector_position[0]**2 + end_effector_position[1]**2)
        
        # Strong penalty for staying near the base
        if base_xy_distance < 0.25:  # Close to base in XY plane
            reward -= 0.5  # Increased penalty for staying near base
        
        # Check for movement (discourage staying still)
        if len(self.position_history) > 10:
            # Calculate variance of positions to detect movement
            positions = np.array(self.position_history[-10:])
            position_variance = np.var(positions, axis=0).sum()
            
            # If very little movement, apply penalty
            if position_variance < 0.0001:
                reward -= 0.2
        
        # Different rewards based on current phase
        if not self.reached_container:
            # Phase 1: Go to container
            
            # Continuous distance-based reward (stronger gradient)
            normalized_distance = min(1.0, container_distance / 1.5)  # Normalize to [0,1]
            distance_reward = (1.0 - normalized_distance) * 0.5  # Increased from 0.2 to 0.5
            reward += distance_reward
            
            # Progress reward (getting closer than before)
            if container_distance < self.best_container_distance:
                reward += 1.0 * (self.best_container_distance - container_distance)  # Doubled from 0.5 to 1.0
            
            # Extra reward for being close to container
            if container_distance < 0.5:
                reward += 0.5  # Extra reward for getting close
            
            # Check if reached container
            if container_distance < self.success_threshold:
                reward += 100.0  # Increased from 50 to 100
                self.reached_container = True
                print(f"Reached {self.target_name} container! Distance: {container_distance:.3f}")
        else:
            # Phase 2: Go to cup
            
            # Continuous distance-based reward (stronger gradient)
            normalized_distance = min(1.0, cup_distance / 1.5)  # Normalize to [0,1]
            distance_reward = (1.0 - normalized_distance) * 0.5  # Increased from 0.2 to 0.5
            reward += distance_reward
            
            # Progress reward (getting closer than before)
            if cup_distance < self.best_cup_distance:
                reward += 1.0 * (self.best_cup_distance - cup_distance)  # Doubled from 0.5 to 1.0
            
            # Extra reward for being close to cup
            if cup_distance < 0.5:
                reward += 0.5  # Extra reward for getting close
            
            # Check if reached cup
            if cup_distance < self.success_threshold:
                reward += 200.0  # Increased from 100 to 200
                print(f"Successfully poured {self.target_name}! Distance: {cup_distance:.3f}")
                return reward, True  # Task complete
        
        # Penalty for being far away from both targets
        if container_distance > 0.6 and cup_distance > 0.6:
            reward -= 0.5
        
        return reward, False
    
    def close(self):
        """Close environment"""
        self.env.close()
    
    def advance_curriculum(self):
        """Advance to the next curriculum phase"""
        if self.curriculum_phase < 2:
            self.curriculum_phase += 1
            print(f"Advancing to curriculum phase {self.curriculum_phase}")
            self._apply_curriculum()