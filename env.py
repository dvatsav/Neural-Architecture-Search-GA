import logging
import torch
from torch.utils.tensorboard import SummaryWriter


log_format = '%(asctime)s %(message)s'
logging.basicConfig(filename="logs.txt", level=logging.INFO, filemode='a',
					format=log_format, datefmt='%m/%d %I:%M:%S %p')

writer = SummaryWriter('runs')

random_seed = 1
torch.manual_seed(random_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print ("Device:", device)
logging.info("Using device %s", str(device))