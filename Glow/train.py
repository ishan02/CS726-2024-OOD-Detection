import os
import random
from itertools import islice
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import logging
import torch.backends.cudnn as cudnn
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage, Loss

# Get the parent directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to the Python path
sys.path.append(parent_dir)

from build_dataset import build_or_get_dataset, get_dataloader
from model import Glow
from config import get_config

config = get_config()
log_dir = './Glow/logs'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(config['output_dir'], exist_ok=True)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(f"{log_dir}/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Log configuration settings
logger.info("Training configuration:")
for key, value in config.items():
    logger.info(f"{key}: {value}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using Device: {device}")

trainset, testset, classes = build_or_get_dataset('cifar10', root='../data',task_generation=True)
trainloader = get_dataloader(trainset, batch_size = config["batch_size"], drop_last=True)
testloader = get_dataloader(testset, batch_size = config["batch_size"], shuffle= False)
num_classes = len(classes)

cudnn.benchmark = True

random.seed(config['seed'])
torch.manual_seed(config['seed'])
logger.info("Using seed: {seed}".format(seed=config['seed']))

multi_class = False
image_shape = (32, 32, 3)

model = Glow(image_shape,config['hidden_channels'],config['K'],config['L'],config['actnorm_scale'],
    config['flow_permutation'],config['flow_coupling'],config['LU_decomposed'],num_classes,config['learn_top'],config['y_condition'],
).to(device)
optimizer = optim.Adamax(model.parameters(), lr=config['lr'], weight_decay=5e-5)
lr_lambda = lambda epoch: min(1.0, (epoch + 1) / config['warmup'])  # noqa
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

def compute_loss(nll, reduction="mean"):
    if reduction == "mean":
        losses = {"nll": torch.mean(nll)}
    elif reduction == "none":
        losses = {"nll": nll}

    losses["total_loss"] = losses["nll"]

    return losses


def compute_loss_y(nll, y_logits, y_weight, y, multi_class, reduction="mean"):
    if reduction == "mean":
        losses = {"nll": torch.mean(nll)}
    elif reduction == "none":
        losses = {"nll": nll}

    if multi_class:
        y_logits = torch.sigmoid(y_logits)
        loss_classes = F.binary_cross_entropy_with_logits(
            y_logits, y, reduction=reduction
        )
    else:
        loss_classes = F.cross_entropy(
            y_logits, torch.argmax(y, dim=1), reduction=reduction
        )

    losses["loss_classes"] = loss_classes
    losses["total_loss"] = losses["nll"] + y_weight * loss_classes

    return losses

def step(engine, batch):
    model.train()
    optimizer.zero_grad()

    x, y = batch
    x = x.to(device)

    if config['y_condition']:
        y = y.to(device)
        z, nll, y_logits = model(x, y)
        losses = compute_loss_y(nll, y_logits, config['y_weight'], y, multi_class)
    else:
        z, nll, y_logits = model(x, None)
        losses = compute_loss(nll)

    losses["total_loss"].backward()

    if config['max_grad_clip'] > 0:
        torch.nn.utils.clip_grad_value_(model.parameters(), config['max_grad_clip'])
    if config['max_grad_norm'] > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])

    optimizer.step()

    return losses

def eval_step(engine, batch):
    model.eval()

    x, y = batch
    x = x.to(device)

    with torch.no_grad():
        if config['y_condition']:
            y = y.to(device)
            z, nll, y_logits = model(x, y)
            losses = compute_loss_y(
                nll, y_logits, config['y_weight'], y, multi_class, reduction="none"
            )
        else:
            z, nll, y_logits = model(x, None)
            losses = compute_loss(nll, reduction="none")

    return losses

trainer = Engine(step)
checkpoint_handler = ModelCheckpoint(
    config['output_dir'], "glow", n_saved=2, require_empty=False
)

trainer.add_event_handler(
    Events.EPOCH_COMPLETED,
    checkpoint_handler,
    {"model": model, "optimizer": optimizer},
)

monitoring_metrics = ["total_loss"]
RunningAverage(output_transform=lambda x: x["total_loss"]).attach(
    trainer, "total_loss"
)

evaluator = Engine(eval_step)

# Note: replace by https://github.com/pytorch/ignite/pull/524 when released
Loss(
    lambda x, y: torch.mean(x),
    output_transform=lambda x: (
        x["total_loss"],
        torch.empty(x["total_loss"].shape[0]),
    ),
).attach(evaluator, "total_loss")

if config['y_condition']:
    monitoring_metrics.extend(["nll"])
    RunningAverage(output_transform=lambda x: x["nll"]).attach(trainer, "nll")

    # Note: replace by https://github.com/pytorch/ignite/pull/524 when released
    Loss(
        lambda x, y: torch.mean(x),
        output_transform=lambda x: (x["nll"], torch.empty(x["nll"].shape[0])),
    ).attach(evaluator, "nll")

pbar = ProgressBar()
pbar.attach(trainer, metric_names=monitoring_metrics)

# load pre-trained model if given
if config['saved_model']:
    model.load_state_dict(torch.load(config['saved_model']))
    model.set_actnorm_init()

    if config['saved_optimizer']:
        optimizer.load_state_dict(torch.load(config['saved_optimizer']))

    file_name, ext = os.path.splitext(config['saved_model'])
    resume_epoch = int(file_name.split("_")[-1])

    @trainer.on(Events.STARTED)
    def resume_training(engine):
        engine.state.epoch = resume_epoch
        engine.state.iteration = resume_epoch * len(engine.state.dataloader)

@trainer.on(Events.STARTED)
def init(engine):
    model.train()

    init_batches = []
    init_targets = []

    with torch.no_grad():
        for batch, target in islice(trainloader, None, config['n_init_batches']):
            init_batches.append(batch)
            init_targets.append(target)

        init_batches = torch.cat(init_batches).to(device)

        assert init_batches.shape[0] == config['n_init_batches'] * config['batch_size']

        if config['y_condition']:
            init_targets = torch.cat(init_targets).to(device)
        else:
            init_targets = None

        model(init_batches, init_targets)

@trainer.on(Events.EPOCH_COMPLETED)
def evaluate(engine):
    evaluator.run(testloader)

    scheduler.step()
    metrics = evaluator.state.metrics

    losses = ", ".join([f"{key}: {value:.2f}" for key, value in metrics.items()])

    logger.info(f"Validation Results - Epoch: {engine.state.epoch} {losses}")

timer = Timer(average=True)
timer.attach(
    trainer,
    start=Events.EPOCH_STARTED,
    resume=Events.ITERATION_STARTED,
    pause=Events.ITERATION_COMPLETED,
    step=Events.ITERATION_COMPLETED,
)

@trainer.on(Events.EPOCH_COMPLETED)
def print_times(engine):
    pbar.log_message(
        f"Epoch {engine.state.epoch} done. Time per batch: {timer.value():.3f}[s]"
    )
    logger.info(f"Epoch {engine.state.epoch} done. Time per batch: {timer.value():.3f}[s]")
    timer.reset()  
trainer.run(trainloader, config['epochs'])

