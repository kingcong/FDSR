import numpy as np
from PIL import Image

class dataset_G():
    """NYUDataset."""

    def __init__(self, root_dir, scale=4, train=True, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            scale (float): dataset scale
            train (bool): train or test
            transform (callable, optional): Optional transform to be applied on a sample.
            
        """
        self.root_dir = root_dir

        self.scale = scale
        
        if train:
            self.depths = np.load('%s/train_depth_split.npy'%root_dir)
            self.images = np.load('%s/train_images_split.npy'%root_dir)
        else:
            self.depths = np.load('%s/test_depth.npy'%root_dir)
            self.images = np.load('%s/test_images_v2.npy'%root_dir)

    def __len__(self):
        return self.depths.shape[0]

    def __getitem__(self, idx):
        depth = self.depths[idx]
        image = self.images[idx]

        h, w = depth.shape

        s = self.scale
        target = np.array(Image.fromarray(depth).resize((w//s,h//s),Image.BICUBIC).resize((w, h), Image.BICUBIC))

        depth = np.expand_dims(depth,2)
        depth=depth*255.0
        target = np.expand_dims(target,2)
        target=target*255.0 
      
        sample = {'guidance': image, 'target': target, 'gt': depth}
        return image, target, depth
