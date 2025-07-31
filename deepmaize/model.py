import torch
import torch.nn as nn
import timm

class SwinMaizeClassifier(nn.Module):
    """
    Maize leaf disease classifier using Swin Transformer architecture.
    """
    def __init__(self, num_classes=4, model_name='swin_base_patch4_window7_224'):
        super().__init__()
        
        # Load base model
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            global_pool=''
        )
        
        # Get number of features
        n_features = self.model.num_features
        
        # Create custom classifier head
        self.classifier = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # Get features from Swin
        x = self.model.forward_features(x)  # [B, H, W, C]
        
        # Global average pooling over spatial dimensions
        x = x.mean(dim=[1, 2])  # Take mean over H and W dimensions
        
        # Classification
        x = self.classifier(x)
        
        return x

class HybridMaizeClassifier(nn.Module):
    """
    A hybrid maize leaf disease classifier combining Swin Transformer with a CNN backbone 
    (e.g., EfficientNet or ResNet) for enhanced feature representation.

    This model leverages two pretrained backbones:
        - Swin Transformer: captures long-range spatial dependencies
        - CNN (EfficientNet or ResNet): captures local texture and shape features

    Features from both models are concatenated and passed through a custom classifier
    to predict one of the disease categories:
    [Common Rust, Gray Leaf Spot, Blight, Healthy].

    Args:
        num_classes (int): Number of output classes. Default is 4.
        swin_model_name (str): Timm model name for Swin Transformer.
        cnn_model_name (str): Timm model name for the CNN backbone (EfficientNet/ResNet).
    """
    def __init__(self, num_classes=4,
                 swin_model_name='swin_base_patch4_window7_224',
                 cnn_model_name='resnet50'):
        super().__init__()

        # Swin Transformer (no classification head)
        self.swin = timm.create_model(
            swin_model_name, pretrained=True,
            num_classes=0, global_pool=''
        )
        swin_feat_dim = self.swin.num_features

        # CNN Backbone (no classification head)
        self.cnn = timm.create_model(
            cnn_model_name, pretrained=True,
            num_classes=0, global_pool='avg'
        )
        cnn_feat_dim = self.cnn.num_features

        # Final classifier head
        self.classifier = nn.Sequential(
            nn.Linear(swin_feat_dim + cnn_feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Swin features (output shape: [B, H, W, C])
        swin_feats = self.swin.forward_features(x)
        swin_feats = swin_feats.mean(dim=[1, 2])  # Global avg pool over H, W -> [B, C]

        # CNN features (already pooled) -> [B, C]
        cnn_feats = self.cnn(x)

        # Concatenate both feature vectors
        fused_feats = torch.cat((swin_feats, cnn_feats), dim=1)

        # Classification
        return self.classifier(fused_feats)
