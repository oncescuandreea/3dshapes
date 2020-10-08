import torch.nn as nn
from base import BaseModel

class ShapeModelRetrievalText(BaseModel):
    def __init__(self, no_inputs):
        super().__init__()
        self.fc1_text = nn.Linear(no_inputs, 300)
        self.fc2_text = nn.Linear(300, 150)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        x = self.fc1_text(x)
        x = self.fc2_text(x)
        return x
