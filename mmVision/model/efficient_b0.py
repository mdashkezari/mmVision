
import torch
import torchvision


class Efficient_b0():
    def __init__(self,
                 n_classes: int,
                 ) -> None:
        """

        Parameters
        -------------
        n_classes: int
            Number of classes.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_classes = n_classes
        self.model_name = "efficient_b0"

        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        self.model = torchvision.models.efficientnet_b0(weights=weights).to(self.device)
        self.transformer = weights.transforms()

        for param in self.model.features.parameters():
            param.requires_grad = False
        self.model.classifier = torch.nn.Sequential(
                                                    torch.nn.Dropout(p=0.2, inplace=True), 
                                                    torch.nn.Linear(in_features=1280, 
                                                                    out_features=self.n_classes,
                                                                    bias=True)).to(self.device)
        return

 