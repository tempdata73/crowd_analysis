import sys
import json
import logging
import argparse
import torch
import numpy as np

from skimage.measure import compare_ssim as ssim
from utils.model import CSRNet
from utils.augmentation import create_dataloader


def main(args):
    # Metrics used for evaluation
    metrics = {'mae': 0, 'mse': 0, 'ssim': 0}
    # Use either CPU or GPU (the later is preferable)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info('Evaluating using {} device'.format(device))

    # Create dataset
    with open(args.test_json) as infile:
        image_paths = json.load(infile)
    test_loader = create_dataloader(
        image_paths, augment=False, batch_size=1, shuffle=True)
    logging.info('Evaluating on {} images'.format(len(test_loader)))

    # Load model
    logging.info('Building model')
    ckpt = torch.load(args.model_path, map_location=torch.device(device))
    model = CSRNet(training=False)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)  # Connect to gpu or cpu

    # Test the model using MSE and MAE
    logging.info('Starting evaluation')
    model.eval()
    for i, (image, target) in enumerate(test_loader):
        # Fix target type and add batch_dimension
        target = target.type(torch.FloatTensor).unsqueeze(0)
        # Connect to either cpu or gpu
        image = image.to(device)
        target = target.to(device)
        # Create density map
        output = model(image)
        # Fetch loss
        instance_mae = torch.abs(output.sum() - target.sum())
        instance_mse = (output.sum() - target.sum()) ** 2
        # Calculate SSIM
        # Even when working with GPUs, PyTorch can't convert
        # tensors to numpy while using CUDA
        ssim_target = target.squeeze().cpu().numpy()
        ssim_output = output.detach().squeeze().cpu().numpy()
        instance_ssim = ssim(ssim_target, ssim_output)
        # Update metrics
        metrics['mae'] += instance_mae.item()
        metrics['mse'] += instance_mse.item()
        metrics['ssim'] += instance_ssim

        # Log every n
        if i % 100 == 0:
            image_info = 'Image:{}'.format(i)
            instance_info = ':Error:{:0.4f}:SquaredError:{:0.4f}'.format(
                instance_mae, instance_mse)
            avg_info = '\tMAE = {:0.4f}\tMSE = {:0.4f}'.format(
                metrics['mae'] / (i + 1), np.sqrt(metrics['mse'] / (i + 1)))
            logging.info(image_info + instance_info + avg_info)

    # Obtain average
    metrics['ssim'] /= len(test_loader)
    metrics['mae'] /= len(test_loader)
    metrics['mse'] = np.sqrt(metrics['mse'] / len(test_loader)).item()

    # Save metrics
    with open('data/metrics.json', 'w') as outfile:
        json.dump(metrics, outfile)
        logging.info('Metrics saved in data/metrics.json')


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='Trained model file')
    parser.add_argument('test_json', type=str, help='Image paths to test on')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(parse_arguments(sys.argv[1:]))
