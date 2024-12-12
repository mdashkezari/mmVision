
import torch
from torch import nn

class TinyVGG(nn.Module):
    def __init__(self,
                 image_size: tuple[int, int],
                 channels: int,
                 hidden_units: int,
                 n_classes: int
                 ) -> None:
        """

        Parameters
        ------------
        channels: int
            Number of channels in the input image.
        

        """
        super().__init__()
        self.model_name = "tinyVGG"
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, 
                      out_channels=hidden_units, 
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * int(image_size[0] * image_size[1] / 16),
                      out_features=n_classes)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x




# from torchvision.transforms import v2
# transformer = v2.Compose([
#                    v2.ToImage(),
#                    v2.ToDtype(torch.uint8, scale=True),
#                    v2.Resize(self.image_size),
#                    v2.ToDtype(torch.float32, scale=True),
#                    ])

# model = TinyVGG(
#         image_size=self.image_size,
#         channels=self.color_channel,
#         hidden_units=10,
#         n_classes=len(self.train_dataset.classes)
#         )