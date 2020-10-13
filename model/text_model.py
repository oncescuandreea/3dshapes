import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

class ShapeModelRetrievalText(BaseModel):
    def __init__(self, no_inputs):
        super().__init__()
        self.fc1_text = nn.Linear(no_inputs, 300)
        self.fc2_text = nn.Linear(300, 150)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        # x = F.relu(self.fc1(x))
        x = self.fc1_text(x)
        x = self.fc2_text(x)
        return x

class ShapeModelRetrievalTextEmb(BaseModel):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(37, 150)
        self.fc1_text = nn.Linear(900, 150)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        x = self.fc1_text(x)
        return x