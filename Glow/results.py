import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import os
import sys
import seaborn as sns
# Get the parent directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to the Python path
sys.path.append(parent_dir)


from build_dataset import build_or_get_dataset, get_dataloader
from model import Glow

device = torch.device("cuda")
from config import get_config

config = get_config()
path = "./Glow/checkpoints/glow_checkpoint_5467.pt"

_, test_cifar, classes = build_or_get_dataset('cifar10', root='../data',task_generation=True)
_, test_svhn, classes = build_or_get_dataset('svhn', root='../data',task_generation=True)


num_classes = 10
image_shape = (32, 32, 3)
model = Glow(image_shape,config['hidden_channels'],config['K'],config['L'],config['actnorm_scale'],
    config['flow_permutation'],config['flow_coupling'],config['LU_decomposed'],num_classes,config['learn_top'],config['y_condition'],
)
model.load_state_dict(torch.load(path)['model'])
model.set_actnorm_init()
model = model.to(device)
model = model.eval()

def compute_nll(dataset, model):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024)
    nlls = []
    for x,y in dataloader:
        x = x.to(device)
        if config['y_condition']:
            y = y.to(device)
        else:
            y = None
        with torch.no_grad():
            _, nll, _ = model(x, y_onehot=y)
            nlls.append(nll)   
    return torch.cat(nlls).cpu()
cifar_nll = compute_nll(test_cifar, model)
svhn_nll = compute_nll(test_svhn, model)

print("CIFAR NLL", torch.mean(cifar_nll))
print("SVHN NLL", torch.mean(svhn_nll))

plt.figure(figsize=(20,10))
plt.title("Histogram Glow - trained on CIFAR10")
plt.xlabel("Negative bits per dimension OR log p(x)")
plt.hist(-svhn_nll.numpy(), label="SVHN", density=True, bins=200)
plt.hist(-cifar_nll.numpy(), label="CIFAR10", density=True, bins=200)
plt.legend()
plt.savefig("./Glow/fig/histogram_glow_cifar_svhn.png", dpi=300)
plt.show()






