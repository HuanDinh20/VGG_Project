from platform import architecture

import torch

from module import VGG
import utils
from train import per_epoch_activity
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# 1. Define some parameters
batch_size = 4
EPOCH = 2
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_path = 'save_model/trained_model/VGG_01.pth'
root_path = r'A:\huan_shit\Study_Shit\Deep_Learning\Side_Projects\LeNet_project\data'
architecture = [[1, (1, 64)], [1, (64, 128)], [1, (128, 256)], [2, (256, 256)], [1, (256, 512)], [2, (512, 512)],
                [2, (512, 512)]]
device = utils.get_device()
summary_writer = SummaryWriter()
# 2. GET DATA

train_aug, test_aug = utils.image_augmentation()
train_set, test_set = utils.get_FashionMNIST(root_path, train_aug, test_aug)
train_loader, test_loader = utils.create_data_loader(train_set, test_set, batch_size)

# 3. model
model = VGG(architecture, 10)
model.to(device)

# 4. loss function
loss_fn = torch.nn.CrossEntropyLoss()

# 5. optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

if __name__ == '__main__':
    per_epoch_activity(train_loader, test_loader, device, optimizer, model, loss_fn, summary_writer, EPOCH, timestamp)
    torch.save(model, model_path)