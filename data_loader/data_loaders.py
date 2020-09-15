from torchvision import transforms
from base import BaseDataLoader
from model.dataset import Shapes3d
from collections import Counter


class ShapeLoader(BaseDataLoader):
    """
    Data loader for 3d-shape dataset
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0,
                 num_workers=1, training=True, test_split=0.1):
        trsfm = transforms.Compose([
            transforms.ToTensor()
            # transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = Shapes3d(self.data_dir, transform=trsfm,
                                train=training, test_split=test_split)
        # import pdb; pdb.set_trace()
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
    
    # def stats_of_data(self):
    #     labels = self.dataset[:][1]
    #     no_tasks = len(labels[0])
    #     list_of_counters = []
    #     for i in range(0, no_tasks):
    #         list_of_counters.append(Counter(labels[:, i]))
    #     return list_of_counters
        
