import os
import cv2
import h5py
import logging
import numpy as np

from scipy.io import loadmat
from scipy.spatial import KDTree
from scipy.ndimage.filters import gaussian_filter


class Base(object):

    def __init__(self, images_dir, anns_dir, save_dir):
        self.images_dir = images_dir
        self.anns_dir = anns_dir
        self.save_dir = save_dir

        if not os.path.exists(self.save_dir):
            logging.info('Creating {} directory'.format(self.save_dir))
            os.mkdir(self.save_dir)

    def save_density_map(self, filename, density):
        dst_path = os.path.join(self.save_dir, filename)
        # h5py saves images more efficiently
        with h5py.File(dst_path, 'w') as outfile:
            outfile['density'] = density

    def load_groundtruth(self, annotations_path):
        pass

    def generate_density_map(self, impulse):
        pass

    def _obtain_annotations_filenames(self, image_filenames):
        return os.listdir(self.anns_dir)

    def generate_impulse(self, groundtruth, size):
        # Apply Dirac's delta function for impulse generation
        # on head annotations
        impulse = np.zeros(size)
        for x, y in groundtruth:
            if x < size[1] and y < size[0]:
                impulse[y, x] = 1
        return impulse

    def create_groundtruth(self):
        image_filenames = os.listdir(self.images_dir)
        annotations_filenames = self._obtain_annotations_filenames(image_filenames)

        for i, (image_fn, anns_fn) in enumerate(
                zip(image_filenames, annotations_filenames)):

            # Check files are correctly matched
            logging.debug('Image_fn: {}'.format(image_fn))
            logging.debug('Anns_fn: {}'.format(anns_fn))
            if i % 100 == 0:
                logging.info('Processed {} images'.format(i))

            # Fetch data
            image_path = os.path.join(self.images_dir, image_fn)
            anns_path = os.path.join(self.anns_dir, anns_fn)
            image = cv2.imread(image_path)
            groundtruth = self.load_groundtruth(anns_path)

            # Create density maps
            impulse = self.generate_impulse(groundtruth, image.shape[:2])
            density = self.generate_density_map(impulse)
            if density is None:
                logging.warning(
                    'Could not create density map for {}'.format(image_path))
                continue

            # There's an accumulated error when creating density plot
            logging.debug(
                'Actual number of people = {}'.format(groundtruth.shape[0]))
            logging.debug(
                'Empirical number of people = {}'.format(density.sum()))

            # Save image
            self.save_density_map(image_fn.replace('.jpg', '.h5'), density)


class Shanghai(Base):

    def __init__(self, root_dir, save_dir):
        images_dir = os.path.join(root_dir, 'images')
        anns_dir = os.path.join(root_dir, 'ground_truth')
        super().__init__(images_dir, anns_dir, save_dir)

    def load_groundtruth(self, annotations_path):
        data = loadmat(annotations_path)
        groundtruth = data['image_info'][0, 0][0, 0][0].astype(np.uint32)
        return groundtruth

    def _obtain_annotations_filenames(self, image_filenames):
        annotations_filenames = []
        for image_fn in image_filenames:
            anns_fn = image_fn.replace('.jpg', '.mat').replace('IMG', 'GT_IMG')
            annotations_filenames.append(anns_fn)
        return annotations_filenames


class Shanghai_A(Shanghai):

    def __init__(self, root_dir, save_dir):
        super().__init__(root_dir, save_dir)

    def generate_density_map(self, impulse, beta=0.3):
        return adaptive_gaussian_filter(impulse, beta=beta)


class Shanghai_B(Shanghai):

    def __init__(self, root_dir, save_dir):
        super().__init__(root_dir, save_dir)

    def generate_density_map(self, impulse):
        return static_gaussian_filter(impulse)


def adaptive_gaussian_filter(impulse, beta=0.3):
    sigmas = []
    density = np.zeros(impulse.shape, dtype=np.float32)
    counts = np.count_nonzero(impulse)

    # No object of interest is in image
    if counts == 0:
        logging.debug('Image did not have any annotations')
        return density

    # Fetch head annotations
    coords = np.nonzero(impulse)
    points = np.array(list(zip(coords[1], coords[0])))  # DEBUG LATER
    # Use KDTree to find top nearest neighbors
    leafsize = 2048
    tree = KDTree(points.copy(), leafsize=leafsize)
    distances, locations = tree.query(points, k=4)

    # Start generating densities
    for i, (x, y) in enumerate(points):
        point_density = np.zeros(impulse.shape, dtype=np.float32)
        point_density[y, x] = 1

        # Select sigma based on nearest neighbors and update density
        if counts > 1:
            sigma = distances[i][1:].mean() * beta
        else:
            sigma = np.mean(impulse.shape) / 4

        density += gaussian_filter(point_density, sigma, mode='constant')
        sigmas.append(sigma)

    return density


def static_gaussian_filter(impulse, sigma=15):
    # Dirac's delta function convolved with static gaussian filter
    return gaussian_filter(impulse, sigma, mode='constant')
