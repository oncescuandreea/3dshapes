import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class ShapeModel(BaseModel):
    def __init__(self, num_tasks=6):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 40, kernel_size=4)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1000, 150)
        self.objs = []
        self.num_tasks = num_tasks
        _NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10,
                                  'object_hue': 10, 'scale': 8,
                                  'shape': 4, 'orientation': 15}
        for i in range(0, num_tasks):
            self.objs.append(nn.Linear(150,
                                       list(_NUM_VALUES_PER_FACTOR.values())[i]))
        self.objs = nn.ModuleList(self.objs)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv3(x)), 2))
        x = x.view(-1, 1000)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        out_objs = []
        for obj in self.objs:
            out_objs.append(F.log_softmax(obj(x), dim=1))
        # x = self.fc2(x)
        # return F.log_softmax(x, dim=1)
        return out_objs
