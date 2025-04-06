import torch
import torch.nn as nn
import timm

class Model(nn.Module):
    def __init__(self, num_classes=1):  # Binary classification = 1 output node
        super().__init__()

        # Load pretrained LeViT backbone (exclude default classifier head)
        self.levit = timm.create_model('levit_128s', pretrained=True, num_classes=num_classes)
        

        # Optionally freeze backbone
        freeze_backbone = True
        if freeze_backbone:
            
            for param in self.levit.parameters():
                param.requires_grad = False
        else:
            print("LeViT backbone parameters will be trainable (fine-tuning).")

        # LeViT-128s returns 384 features per token
        levit_out_features = self.levit.num_features  # Expected to be 384

        # 1x1 Conv layer to reduce features (optional)
        intermediate_features = 64
        self.conv_after_levit = nn.Conv2d(levit_out_features, intermediate_features, kernel_size=1)
        self.relu = nn.ReLU()

        # Global average pooling to [B, C, 1, 1]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Final classifier head
        self.head = nn.Linear(intermediate_features, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        

        # Extract features (token-based output: [B, 16, 384])
        x = self.levit.forward_features(x)
        

        # Convert token-based output to a 4D feature map
        B, N, D = x.shape  # N should be 16 and D is 384
        # Transpose to [B, 384, 16]
        x = x.transpose(1, 2)
        # Reshape to [B, 384, 4, 4] assuming 16 tokens can form a 4x4 grid
        x = x.view(B, D, int(N ** 0.5), int(N ** 0.5))
        

        # Optional conv layer
        x = self.conv_after_levit(x)       # [B, 64, 4, 4]
        
        x = self.relu(x)

        # Pool: reduce spatial dimensions to [B, 64, 1, 1]
        x = self.pool(x)
        

        # Flatten: [B, 64]
        x = x.flatten(1)
        

        # Classifier: [B, num_classes]
        x = self.head(x)
        # x = x.squeeze(1)
        

        return x