import os
from dotenv import load_dotenv
import torch
import numpy as np
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.configs.types import FeatureType, PolicyFeature, NormalizationMode
from lerobot.common.policies.normalize import Normalize, Unnormalize

def prepare_image(image_size=(224, 224)):
    """Create a sample image and normalize to [-1, 1]"""
    # Create random image for testing
    image = np.random.uniform(0, 1, (3, image_size[0], image_size[1])).astype(np.float32)
    image = torch.from_numpy(image)
    # Normalize from [0,1] to [-1,1] as expected by SigLIP
    image = image * 2.0 - 1.0
    return image[None]  # Add batch dimension

def prepare_robot_state(state_dim=6):
    """Create a sample robot state"""
    state = torch.zeros(1, state_dim)  # Batch size 1
    return state

def get_dataset_stats():
    """Create sample dataset statistics for normalization"""
    stats = {
        "observation.state": {
            "min": torch.zeros(6),
            "max": torch.ones(6),
            "mean": torch.zeros(6),
            "std": torch.ones(6),
        },
        "observation.images.cam_high": {
            "min": torch.full((3, 1, 1), -1.0),
            "max": torch.full((3, 1, 1), 1.0),
            "mean": torch.zeros(3, 1, 1),
            "std": torch.ones(3, 1, 1),
        },
        "action": {
            "min": torch.full((6,), -1.0),
            "max": torch.full((6,), 1.0),
            "mean": torch.zeros(6),
            "std": torch.ones(6),
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
        # Load the pretrained policy
        print("Loading PI0 model...")
        policy = PI0Policy.from_pretrained("lerobot/pi0")
        
        # Register the image feature
        policy.config.input_features["observation.images.cam_high"] = PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 224, 224)
        )
        
        # Register the state feature
        policy.config.input_features["observation.state"] = PolicyFeature(
            type=FeatureType.STATE,
            shape=(6,)
        )

        # Register the action feature
        policy.config.output_features["action"] = PolicyFeature(
            type=FeatureType.ACTION,
            shape=(6,)
        )

        # Set normalization modes
        policy.config.normalization_mapping = {
            "VISUAL": NormalizationMode.IDENTITY,  # Images are already normalized
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD
        }
        
        # Create normalizers with updated config
        dataset_stats = get_dataset_stats()
        policy.normalize_inputs = Normalize(policy.config.input_features, policy.config.normalization_mapping, dataset_stats)
        policy.normalize_targets = Normalize(policy.config.output_features, policy.config.normalization_mapping, dataset_stats)
        policy.unnormalize_outputs = Unnormalize(policy.config.output_features, policy.config.normalization_mapping, dataset_stats)
        
        policy = policy.to(device)
        policy.eval()
        print("PI0 model initialized successfully!")
        
        # Prepare sample batch
        batch = {
            "observation.state": prepare_robot_state().to(device),
            "observation.images.cam_high": prepare_image().to(device),
            "task": ["pick up the object\n"]  # Must end with newline
        }
        
        # Get action from policy
        print("Running inference...")
        with torch.no_grad():
            action = policy.select_action(batch)
        
        print("Action shape:", action.shape)
        print("Action values:", action)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 