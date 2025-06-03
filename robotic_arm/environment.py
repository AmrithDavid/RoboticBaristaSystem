import pybullet as p
import pybullet_data
import numpy as np
import time

class RoboticBaristaEnv:
    def __init__(self, gui=True):
        # Connect to the physics server
        self.physics_client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Set camera position for better visualization
        p.resetDebugVisualizerCamera(
            cameraDistance=1.2,
            cameraYaw=30,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.5]
        )
        
        # Load basic scene elements
        self._load_scene()
        
        print("Environment initialised")
    
    def _load_scene(self):
        """Load the basic scene elements"""
        # Load plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Create a simple table
        table_height = 0.5
        table_size = [0.8, 0.8, table_height/2]
        table_position = [0, 0, table_height/2]
        
        # Create collision shape for table
        self.table_collision_id = p.createCollisionShape(
            p.GEOM_BOX, 
            halfExtents=table_size
        )
        
        # Create visual shape for table
        self.table_visual_id = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=table_size, 
            rgbaColor=[0.8, 0.6, 0.4, 1]
        )
        
        # Create table body
        self.table_id = p.createMultiBody(
            baseMass=0,  # Static object
            baseCollisionShapeIndex=self.table_collision_id,
            baseVisualShapeIndex=self.table_visual_id,
            basePosition=table_position
        )
        
        # Store table height for later use
        self.table_height = table_height
        
        # Load robot arm
        self._load_robot_arm()
        
        # Load containers and cup
        self._load_containers()
        
        print("Scene loaded")
    
    def _load_robot_arm(self):
        """Load the robotic arm (KUKA iiwa)"""
        # Position the robot on top of the table
        robot_start_pos = [0, 0, self.table_height]
        
        # Load the KUKA iiwa robot arm
        self.robot_id = p.loadURDF(
            "kuka_iiwa/model.urdf",
            robot_start_pos,
            useFixedBase=1  # Fix the robot base
        )
        
        # Get number of joints
        self.num_joints = p.getNumJoints(self.robot_id)
        
        # Store joint information
        self.joints = []
        self.joint_lower_limits = []
        self.joint_upper_limits = []
        
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            if joint_info[2] == p.JOINT_REVOLUTE:  # Only revolute joints
                self.joints.append(i)
                self.joint_lower_limits.append(joint_info[8])  # Lower limit
                self.joint_upper_limits.append(joint_info[9])  # Upper limit
        
        # Define end effector index (the last joint)
        self.end_effector_index = 6
        
        print(f"Robot arm loaded with {len(self.joints)} controllable joints")
    
    def _load_containers(self):
        """Create containers for coffee and matcha, and a cup"""
        # Container dimensions
        container_size = [0.07, 0.07, 0.1]
        container_half_size = [s/2 for s in container_size]
        
        # Cup dimensions
        cup_radius = 0.05
        cup_height = 0.08
        
        # Container positions - placed further apart
        self.coffee_position = [0.5, 0.5, self.table_height + container_size[2]/2]  # Further away
        self.matcha_position = [0.5, -0.5, self.table_height + container_size[2]/2]  # Further away
        self.cup_position = [0.5, 0, self.table_height + cup_height/2]  # Further away
        
        # Create coffee container (brown)
        self.coffee_collision_id = p.createCollisionShape(
            p.GEOM_BOX, 
            halfExtents=container_half_size
        )
        
        self.coffee_visual_id = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=container_half_size, 
            rgbaColor=[0.6, 0.4, 0.2, 1]  # Brown color
        )
        
        self.coffee_container_id = p.createMultiBody(
            baseMass=0,  # Made static (was 0.2 before)
            baseCollisionShapeIndex=self.coffee_collision_id,
            baseVisualShapeIndex=self.coffee_visual_id,
            basePosition=self.coffee_position
        )
        
        # Create matcha container (green)
        self.matcha_collision_id = p.createCollisionShape(
            p.GEOM_BOX, 
            halfExtents=container_half_size
        )
        
        self.matcha_visual_id = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=container_half_size, 
            rgbaColor=[0.2, 0.8, 0.2, 1]  # Green color
        )
        
        self.matcha_container_id = p.createMultiBody(
            baseMass=0,  # Made static (was 0.2 before)
            baseCollisionShapeIndex=self.matcha_collision_id,
            baseVisualShapeIndex=self.matcha_visual_id,
            basePosition=self.matcha_position
        )
        
        # Create cup (white/transparent)
        self.cup_collision_id = p.createCollisionShape(
            p.GEOM_CYLINDER, 
            radius=cup_radius, 
            height=cup_height
        )
        
        self.cup_visual_id = p.createVisualShape(
            p.GEOM_CYLINDER, 
            radius=cup_radius, 
            length=cup_height, 
            rgbaColor=[1, 1, 1, 0.8]  # White, slightly transparent
        )
        
        self.cup_id = p.createMultiBody(
            baseMass=0,  # Made static (was 0.1 before)
            baseCollisionShapeIndex=self.cup_collision_id,
            baseVisualShapeIndex=self.cup_visual_id,
            basePosition=self.cup_position
        )
        
        print("Containers and cup loaded")
    
    def reset(self):
        """Reset the environment to initial state"""
        # Reset joint positions to home configuration
        for i in self.joints:
            p.resetJointState(self.robot_id, i, 0)
        
        # Allow physics to stabilize
        for _ in range(100):
            p.stepSimulation()
        
        print("Environment reset")
        
        # Return initial observation
        return self._get_observation()
    
    def _get_observation(self):
        """Get current state observation"""
        # Get joint states
        joint_positions = []
        joint_velocities = []
        
        for i in self.joints:
            joint_state = p.getJointState(self.robot_id, i)
            joint_positions.append(joint_state[0])
            joint_velocities.append(joint_state[1])
        
        # Get end effector state
        end_effector_state = p.getLinkState(self.robot_id, self.end_effector_index)
        end_effector_position = end_effector_state[0]  # Position in Cartesian space
        
        # Combine all information into observation vector
        observation = np.array(joint_positions + joint_velocities + list(end_effector_position))
        
        return observation
    
    def step(self, action):
        """Execute one step in the environment"""
        # Apply action (joint velocities)
        for i, joint_idx in enumerate(self.joints):
            if i < len(action):
                p.setJointMotorControl2(
                    self.robot_id,
                    joint_idx,
                    p.VELOCITY_CONTROL,
                    targetVelocity=action[i],
                    force=500  # Maximum force to apply
                )
        
        # Step physics simulation
        p.stepSimulation()
        
        # Get new observation
        observation = self._get_observation()
        
        # Simple placeholder reward
        reward = 0.0
        
        # Episode is not done by default
        done = False
        
        # Additional info dict
        info = {}
        
        return observation, reward, done, info
    
    def move_to_pose(self, target_position, max_steps=1000, threshold=0.05):
        """Move the end-effector to a target position using inverse kinematics"""
        print(f"Moving to position {target_position}")
        step_count = 0
        
        while step_count < max_steps:
            # Get current end effector position
            end_effector_state = p.getLinkState(self.robot_id, self.end_effector_index)
            current_position = end_effector_state[0]
            
            # Calculate distance to target
            distance = np.sqrt(sum([(current_position[i] - target_position[i])**2 for i in range(3)]))
            
            # If close enough, consider it reached
            if distance < threshold:
                print(f"Reached target position in {step_count} steps")
                return True
            
            # Calculate inverse kinematics
            joint_positions = p.calculateInverseKinematics(
                bodyUniqueId=self.robot_id,
                endEffectorLinkIndex=self.end_effector_index,
                targetPosition=target_position
            )
            
            # Set joint positions with lower force and slower movement
            for i, joint_idx in enumerate(self.joints):
                if i < len(joint_positions):
                    # Get current position
                    current = p.getJointState(self.robot_id, joint_idx)[0]
                    # Move only a small step towards target (slow movement)
                    target = joint_positions[i]
                    # Move only 2% of the way each step
                    new_pos = current + (target - current) * 0.02
                    
                    p.setJointMotorControl2(
                        bodyUniqueId=self.robot_id,
                        jointIndex=joint_idx,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=new_pos,
                        force=200  # Lower force for gentler movement
                    )
            
            # Step simulation
            p.stepSimulation()
            time.sleep(0.01)  # Slow down for visualization
            
            step_count += 1
        
        print(f"Failed to reach target position after {max_steps} steps")
        return False
    
    def pour_from_container(self, container_id):
        """Move to a container, then to the cup to simulate pouring"""
        # Select container
        if container_id == 0:
            container_position = self.coffee_position
            container_name = "coffee"
        else:
            container_position = self.matcha_position
            container_name = "matcha"
        
        print(f"\nPreparing {container_name}...")
        
        # Move to a position above the container (not touching)
        above_container = [
            container_position[0], 
            container_position[1],
            container_position[2] + 0.1  # 10cm above container
        ]
        print(f"Moving above {container_name} container")
        success = self.move_to_pose(above_container)
        if not success:
            return False
        
        # Simulate interacting with container
        print(f"Simulating interaction with {container_name} container")
        time.sleep(1)  
        
        # Move to a position above the cup (not touching)
        above_cup = [
            self.cup_position[0],
            self.cup_position[1],
            self.cup_position[2] + 0.1  # 10cm above cup
        ]
        print("Moving above cup")
        success = self.move_to_pose(above_cup)
        if not success:
            return False
        
        # Simulate pouring into cup
        print("Simulating pouring into cup")
        time.sleep(1)
        
        print(f"{container_name.capitalize()} preparation complete!")
        return True
    
    def close(self):
        """Disconnect from the physics server"""
        p.disconnect(self.physics_client)
        print("Environment closed")
