import os
from dotenv import load_dotenv
import torch
import numpy as np
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.configs.types import FeatureType, PolicyFeature, NormalizationMode
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.envs.factory import make_env, make_env_config
from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.constants import OBS_ROBOT, OBS_IMAGES, ACTION

def prepare_image(image_size=(480, 640)):
    """Create a sample image and normalize to [-1, 1]"""
    # Create random image for testing
    image = np.random.uniform(0, 1, (3, image_size[0], image_size[1])).astype(np.float32)
    image = torch.from_numpy(image)
    # Normalize from [0,1] to [-1,1] as expected by SigLIP
    image = image * 2.0 - 1.0
    return image[None]  # Add batch dimension

def prepare_robot_state(state_dim=14):
    """Create a sample robot state"""
    state = torch.zeros(1, state_dim)  # Batch size 1
    return state

def get_dataset_stats():
    """Create sample dataset statistics for normalization"""
    stats = {
        OBS_ROBOT: {  # Using constant instead of hardcoded string
            "min": torch.zeros(14),  # ALOHA has 14 joints
            "max": torch.ones(14),
            "mean": torch.zeros(14),
            "std": torch.ones(14),
        },
        f"{OBS_IMAGES}.top": {  # Using constant instead of hardcoded string
            "min": torch.full((3, 1, 1), -1.0),
            "max": torch.full((3, 1, 1), 1.0),
            "mean": torch.zeros(3, 1, 1),
            "std": torch.ones(3, 1, 1),
        },
        ACTION: {  # Using constant instead of hardcoded string
            "min": torch.full((14,), -1.0),  # ALOHA has 14 joint actions
            "max": torch.full((14,), 1.0),
            "mean": torch.zeros(14),
            "std": torch.ones(14),
        }
    }
    return stats

def main():
    # Load environment variables
    load_dotenv()
    
    # Set Hugging Face token
    hf_token = os.getenv("HUGGING_FACE_TOKEN")
    if hf_token is None:
        raise ValueError("Please set HUGGING_FACE_TOKEN in your .env file")
    os.environ["HUGGING_FACE_TOKEN"] = hf_token
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Create ALOHA simulation environment
        print("Creating ALOHA environment...")
        env_config = make_env_config(env_type="aloha")
        env = make_env(env_config)
        
        # Load PI0 model configured for ALOHA simulation
        print("Loading PI0 model...")
        policy = PI0Policy.from_pretrained(
            "lerobot/pi0",  # We'll use base model since aloha_sim isn't available yet
            dataset_stats=get_dataset_stats()
        )
        
        # Configure for ALOHA simulation
        policy.config.empty_cameras = 2  # Add 2 empty cameras as in pi0_aloha_sim
        policy.config.adapt_to_pi_aloha = True
        policy.config.use_delta_joint_actions_aloha = False
        
        # Get features from environment config
        policy.config.input_features = env_config.features
        policy.config.output_features = {"action": env_config.features["action"]}
        
        # Set normalization modes
        policy.config.normalization_mapping = {
            "VISUAL": NormalizationMode.IDENTITY,  # Images are already normalized
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD
        }
        
        policy = policy.to(device)
        policy.eval()
        print("PI0 model initialized successfully!")
        
        # Run simulation loop
        print("Starting simulation...")
        obs_tuple = env.reset()
        obs = preprocess_observation(obs_tuple[0])  # First element is the observation dict
        done = False
        step = 0
        
        while not done and step < 400:  # Max 400 steps (default ALOHA episode length)
            # Print observation structure to debug
            print("Observation keys:", obs.keys())
            print("State shape:", obs[OBS_ROBOT].shape)
            print("Image shape:", obs[f"{OBS_IMAGES}.top"].shape)
            
            # Prepare inputs for policy
            batch = {
                OBS_ROBOT: obs[OBS_ROBOT].to(device),  # Already has batch dimension
                f"{OBS_IMAGES}.top": obs[f"{OBS_IMAGES}.top"].to(device),  # Already has batch dimension
                "task": ["pick up the object\n"]  # Must end with newline
            }
            
            # Get action from policy
            with torch.no_grad():
                action = policy.select_action(batch)
            
            # Convert action to numpy and execute in env
            action_np = action.cpu().numpy().squeeze()
            obs_tuple = env.step(action_np)
            obs, reward, done, info = obs_tuple  # Unpack the step tuple
            obs = preprocess_observation(obs)  # Preprocess the new observation
            
            step += 1
            if step % 100 == 0:
                print(f"Step {step}")
        
        print(f"Simulation completed after {step} steps")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 