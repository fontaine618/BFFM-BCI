import torch


class EEGNet(torch.nn.Module):
    def __init__(
            self,
            n_channels,
            n_classes,
            droupout_rate=0.5,
            kernel_length=32,
            dim1=4,
            dim2=2,
            dim3=16,
            norm_rate=0.25
    ):
        super(EEGNet, self).__init__()

        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=n_channels,
                out_channels=dim1,
                kernel_size=(1, kernel_length),
                stride=(1, 1),
                padding="same",
                bias=False
            ),
            torch.nn.BatchNorm2d(
                num_features=dim1,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True
            ),

            torch.nn.ELU(alpha=1.0),
            torch.nn.AvgPool2d(kernel_size=(1, 4)),
            torch.nn.Dropout(p=droupout_rate)
        )

        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=dim1,
                out_channels=dim2,
                kernel_size=(1, 16),
                stride=(1, 1),
                padding="same",
                bias=False
            ),
            torch.nn.BatchNorm2d(
                num_features=dim2,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True
            ),
            torch.nn.ELU(alpha=1.0),
            torch.nn.AvgPool2d(kernel_size=(1, 4)),
            torch.nn.Dropout(p=droupout_rate)
        )

        self.classify = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=dim2*60,
                out_features=n_classes,
                bias=True
            ),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        x = self.classify(x)
        return x