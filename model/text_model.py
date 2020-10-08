import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

class ShapeModelRetrievalText(BaseModel):
    def __init__(self, no_inputs):
        super().__init__()
        self.fc1 = nn.Linear(no_inputs, 300)
        self.fc2 = nn.Linear(300, 150)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        # x = F.relu(self.fc1(x))
        x = self.fc1(x)
        x = self.fc2(x)
        return x
