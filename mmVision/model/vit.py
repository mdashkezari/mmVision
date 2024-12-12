
import torch
import torchvision


class VIT():
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

        weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        self.model = torchvision.models.vit_b_16(weights=weights).to(self.device)
        self.transformer = weights.transforms()

        for param in self.model.encoder.parameters():
            param.requires_grad = False
        self.model.heads = torch.nn.Sequential(
                                                    torch.nn.Dropout(p=0.2, inplace=True), 
                                                    torch.nn.Linear(in_features=768, 
                                                                    out_features=self.n_classes,
                                                                    bias=True)).to(self.device)
        return

 