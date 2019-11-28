import logging
import sys
import os
import torch

log_format = '%(asctime)s %(message)s'
logging.basicConfig(filename="logs.txt", level=logging.INFO, filemode='a',
					format=log_format, datefmt='%m/%d %I:%M:%S %p')

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print ("Device:", device)
logging.info("Using device %s", str(device))