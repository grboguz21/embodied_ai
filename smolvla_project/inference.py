from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import numpy as np
from lerobot.common.datasets.utils import write_json, serialize_dict
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.datasets.utils import dataset_to_policy_features
import torch
from PIL import Image
import torchvision
from mujoco_env.y_env2 import SimpleEnv2


from torchvision import transforms
def get_default_transform(image_size: int = 224):
    """
    Returns a torchvision transform that:
     Converts to a FloatTensor and scales pixel values [0,255] -> [0.0,1.0]
    """
    return transforms.Compose([
        transforms.ToTensor(), 
    ])


dataset_metadata = LeRobotDatasetMetadata("omy_pnp_language", root='./omy_pnp_language')
features = dataset_to_policy_features(dataset_metadata.features)

output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
input_features = {key: ft for key, ft in features.items() if key not in output_features}

cfg = SmolVLAConfig(input_features=input_features, output_features=output_features, chunk_size= 5, n_action_steps=5)
delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)

xml_path = './asset/example_scene_y2.xml'
PnPEnv = SimpleEnv2(xml_path, action_type='joint_angle')

step = 0
device = "cuda"
save_image = True
IMG_TRANSFORM = get_default_transform()


policy = SmolVLAPolicy.from_pretrained(
    "Jeongeun/omy_pnp_smolvla", 
    config=cfg, 
    dataset_stats=dataset_metadata.stats
)

PnPEnv.reset(seed=0)
policy.reset()

policy.eval()

while PnPEnv.env.is_viewer_alive():
    PnPEnv.step_env()
    if PnPEnv.env.loop_every(HZ=20):
        # Check if the task is completed
        success = PnPEnv.check_success()
        if success:
            print('Success')
            # Reset the environment and action queue
            policy.reset()
            PnPEnv.reset()
            step = 0
            save_image = False
        # Get the current state of the environment
        state = PnPEnv.get_joint_state()[:6]
        # Get the current image from the environment
        image, wirst_image = PnPEnv.grab_image()
        image = Image.fromarray(image)
        image = image.resize((256, 256))
        image = IMG_TRANSFORM(image)
        wrist_image = Image.fromarray(wirst_image)
        wrist_image = wrist_image.resize((256, 256))
        wrist_image = IMG_TRANSFORM(wrist_image)
        data = {
            'observation.state': torch.tensor([state]).to(device),
            'observation.image': image.unsqueeze(0).to(device),
            'observation.wrist_image': wrist_image.unsqueeze(0).to(device),
            'task': [PnPEnv.instruction],
        }
        # Select an action
        action = policy.select_action(data)
        action = action[0,:7].cpu().detach().numpy()
        # Take a step in the environment
        _ = PnPEnv.step(action)
        PnPEnv.render()
        step += 1
        success = PnPEnv.check_success()
        if success:
            print('Success')
            break