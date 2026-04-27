import torch
import numpy as np
from torch.nn import functional as F
from scipy.ndimage import gaussian_filter


def compute_anomaly_map_from_features(fs_list, ft_list, out_size=224, amap_mode="mul"):
    """
    Aggregates per-layer anomaly maps into a single pixel-wise anomaly map.
    Layer-wise cosine dissimilarities are combined either multiplicatively
    (to emphasize anomalies agreed upon by all layers) or additively
    (to accumulate anomaly evidence across layers).
    """
    # Initialize the anomaly map based on the selected fusion strategy (multiplicative or additive)
    if amap_mode == "mul":
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])

    a_map_list = []

    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]

        # Compute cosine similarity between the feature maps
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)

        # Resize the anomaly map to the target output size using bilinear interpolation
        a_map = F.interpolate(a_map, size=out_size, mode="bilinear", align_corners=True)
        a_map = a_map[0, 0, :, :].to("cpu").detach().numpy()

        a_map_list.append(a_map)

        # Aggregate anomaly evidence across layers
        if amap_mode == "mul":
            anomaly_map *= a_map
        else:
            anomaly_map += a_map

    return anomaly_map, a_map_list


def infer_anomaly_map_from_image(
    encoder,
    proj_layer,
    bn,
    decoder,
    img,
):
    """
    Compute an anomaly map for a single image.

    Pipeline:
        encoder → projection → batch norm → decoder → feature comparison
    """
    # ------------------------------------------------------------
    # Forward pass: feature extraction and reconstruction    
    # ------------------------------------------------------------
    # Extract multi-scale features from the encoder
    feats = encoder(img)
    
    # Project features into latent space
    proj_feats = proj_layer(feats)
    
    # Reconstruct projected features via batch norm and decoder
    decoded_feats = decoder(bn(proj_feats))

    # ------------------------------------------------------------
    # Anomaly map computation    
    # ------------------------------------------------------------
    # Compute and fuse layer-wise anomaly maps into a single pixel-wise map    
    anomaly_map, _ = compute_anomaly_map_from_features(
        feats, decoded_feats, img.shape[-1], amap_mode="a"
    )

    # Smooth the anomaly map spatially using a Gaussian filter
    anomaly_map = gaussian_filter(anomaly_map, sigma=4)

    return anomaly_map


def infer_anomaly_maps_from_dataset(
    *,
    encoder,
    proj_layer,
    bn,
    decoder,
    dataloader,
    device,
):
    """
    Run anomaly map inference over a full dataset.

    Returns list of anomaly maps (one per image)
    """
    encoder.eval()
    proj_layer.eval()
    bn.eval()
    decoder.eval()

    anomaly_maps = []

    with torch.no_grad():
        for img, *_ in dataloader:
            img = img.to(device)
            
            anomaly_maps.append(
                infer_anomaly_map_from_image(
                    encoder,
                    proj_layer,
                    bn,
                    decoder,
                    img,
                )
            )

    return anomaly_maps