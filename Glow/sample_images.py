import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import os
import sys

# Get the parent directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to the Python path
sys.path.append(parent_dir)


from build_dataset import  postprocess
from model import Glow

device = torch.device("cuda")
from config import get_config

config = get_config()
path = "./Glow/checkpoints/glow_checkpoint_10000.pt"


num_classes = 10
image_shape = (32, 32, 3)
model = Glow(image_shape,config['hidden_channels'],config['K'],config['L'],config['actnorm_scale'],
    config['flow_permutation'],config['flow_coupling'],config['LU_decomposed'],num_classes,config['learn_top'],config['y_condition'],
)

model.load_state_dict(torch.load(path)['model'])
model.set_actnorm_init()

model = model.to(device)

model = model.eval()

def sample(model):
    with torch.no_grad():
        if config['y_condition']:

            #y = torch.eye(num_classes)
            y = torch.zeros(10)
            y[9] = 1
            y = y.repeat(35,1)
            #y = y.repeat(48 // num_classes + 1,1)
            y = y[:32, :].to(device) # number hardcoded in model for now
        else:
            y = None

        images = postprocess(model(y_onehot=y, temperature=1, reverse=True))

    return images.cpu()

images = sample(model)
grid = make_grid(images[:30], nrow=6).permute(1,2,0)

plt.figure(figsize=(10,10))
plt.imshow(grid)
plt.savefig('./Glow/fig/class-2.png')
plt.show()
plt.axis('off')