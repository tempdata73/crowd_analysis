import os
import re
import sys
import json
import time
import logging
import argparse
import torch
import torch.nn as nn
import numpy as np

from utils.model import CSRNet
from utils.augmentation import create_dataloader

# Define constants
LR = 1e-6
BATCH_SIZE = 1
MOMENTUM = 0.95
DECAY = 5e-4
START_EPOCH = 0
EPOCHS = 400
PRINT_FREQ = 100

# Create directory to save models
CKPTS_FILE = 'ckpts'
if not os.path.exists(CKPTS_FILE):
    os.mkdir(CKPTS_FILE)


def save_checkpoint(model, optimizer, epoch, loss):
    filename = os.path.join(CKPTS_FILE, 'model-{:0.2f}.pth.tar'.format(loss))
    state = {'epoch': epoch + 1,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, filename)


def load_checkpoint(model, optimizer, filename, device):
    ckpt = torch.load(filename, map_location=device)
    start_epoch = ckpt['epoch']
    model.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
    return model.to(device), optimizer, start_epoch


def main(args):
    global START_EPOCH
    best_pred = None
    loss_data = {'train_mae': [], 'val_mae': []}

    # Fetch training and validation subsets
    with open(args.train_json) as infile:
        train_image_paths = json.load(infile)
    with open(args.val_json) as infile:
        val_image_paths = json.load(infile)

    # Use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info('Using {} device for training'.format(device))

    # Define model
    logging.info('Building model')
    model = CSRNet(training=True).to(device)
    criterion = nn.MSELoss(reduction='sum')
    optim = torch.optim.SGD(
        model.parameters(), LR, momentum=MOMENTUM, weight_decay=DECAY)

    # Continue training after checkpoint
    if args.pretrained:
        logging.info('Loading checkpoint from {}'.format(args.pretrained))
        model, optim, START_EPOCH = load_checkpoint(
            model, optim, args.pretrained, device)
        logging.info('Continue training at epoch {}'.format(START_EPOCH))
        # Replace best_pred with loss of ckpt model
        best_pred = float(re.findall(r'[0-9]*\.[0-9]*', args.pretrained)[0])
        logging.debug('Best pred is {:0.2f}'.format(best_pred))
        # Update metrics
        with open('data/loss_data.json') as infile:
            loss_data = json.load(infile)

    # Training + evaluation
    logging.info('Training model')
    for epoch in range(START_EPOCH, EPOCHS):
        model.train()  # Training mode
        train_loader = create_dataloader(
            train_image_paths, 'validation', batch_size=BATCH_SIZE, shuffle=True)
        # Metrics
        time_info = []
        loss_info = []

        for i, (image, target) in enumerate(train_loader):
            # Make target compatible with output and add batch dimension
            target = target.type(torch.FloatTensor).unsqueeze(0)

            # Transfer to either GPU or CPU
            image = image.to(device)
            target = target.to(device)

            # Zero the parameter gradients
            optim.zero_grad()
            start_time = time.time()
            output = model(image)
            end_time = time.time()
            loss = criterion(output, target)

            # backard + optimize
            loss.backward()
            optim.step()

            # Update metrics
            time_info.append(end_time - start_time)
            loss_info.append(loss.item())

            # Log results
            if i % PRINT_FREQ == 0:
                epoch_text = 'Epoch [{}/{}] ({}/{}) '.format(
                    epoch, EPOCHS, i, len(train_loader))
                time_text = 'Time = {:0.2f}, Total time = {:0.2f} '.format(
                    time_info[-1], np.sum(time_info))
                loss_text = 'Current loss = {:0.3f}, Avg loss = {:0.3f}'.format(
                    loss_info[-1], np.mean(loss_info))
                logging.info(epoch_text + time_text + loss_text)

        logging.info('Evaluating model...')
        model.eval()  # Evaluation mode
        val_loader = create_dataloader(
            val_image_paths, 'validation', batch_size=BATCH_SIZE, shuffle=True)

        mae = 0
        for image, target in val_loader:
            # Make target compatible with output and add batch dimension
            target = target.type(torch.FloatTensor).unsqueeze(0)

            # Transfer to either GPU or CPU
            image = image.to(device)
            target = target.to(device)
            # Create density map
            output = model(image)

            # Calculate MAE without messing with criterion for training
            mae += abs((output.sum() - target.sum()).item())
        # Average out MAE
        mae /= len(val_loader)
        logging.info('Mean average Error (MAE) = {:0.4f}'.format(mae))

        # Save checkpoint if current state beats the best one
        if best_pred is None or mae < best_pred:
            save_checkpoint(model, optim, epoch, mae)
            logging.info('Checkpoint created')
            best_pred = mae

        # Update metrics
        loss_data['train_mae'].append(np.mean(loss_info))
        loss_data['val_mae'].append(mae)

        # Save data
        with open('data/loss_data.json', 'w') as outfile:
            logging.info('Saving loss data in data/loss_data.json')
            json.dump(loss_data, outfile)

    # Save last model
    torch.save(model.state_dict(), os.path.join(CKPTS_FILE, 'model.pth.tar'))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('train_json', type=str, help='Path to train.json')
    parser.add_argument('val_json', type=str, help='Path to val.json')
    parser.add_argument('--pretrained', '-p', type=str, default=None,
                        help='Continue training after checkpoint with model.pth.tar')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(parse_arguments(sys.argv[1:]))
