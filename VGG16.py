import numpy as np
import torch
import torch.nn as nn

class VGG16(nn.Module):
    
    def __init__(self, num_classes=10) -> None:
        super(VGG16,self).__init__()
        
        # n-2p+k/s+1 => 224 - 2*1 +3/1 +1 = 226 * 226 * 64 => 3268864
        self.l1_1 = nn.Sequential( #1
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.l1_2 = nn.Sequential( #2
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        
        self.l2_1 = nn.Sequential( #3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.l2_2 = nn.Sequential( #4
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        
        self.l3_1 = nn.Sequential( #5
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.l3_2 = nn.Sequential( #6
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.l3_3 = nn.Sequential( #7
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        
        self.l4_1 = nn.Sequential( #8
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.l4_2 = nn.Sequential( #9
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        self.l4_3 = nn.Sequential( #10
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        
        self.l5_1 = nn.Sequential( #11
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        self.l5_2 = nn.Sequential( #12
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        self.l5_3 = nn.Sequential( #13
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        
        
        self.fc1 = nn.Sequential( #14
            nn.Dropout(0.5),
            nn.Linear(7*7*512, 4096),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential( #15
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential( #16
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        out = self.l1_1(x)
        out = self.l1_2(out)
        out = self.l2_1(out)
        out = self.l2_2(out)
        out = self.l3_1(out)
        out = self.l3_2(out)
        out = self.l3_3(out)
        out = self.l4_1(out)
        out = self.l4_2(out)
        out = self.l4_3(out)
        out = self.l5_1(out)
        out = self.l5_2(out)
        out = self.l5_3(out)
        out = out.reshape(out.size(0), -1)  # (batch,  height, width, n_chanel) => (64,512,7,7) , (64,512*7*7)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out
        