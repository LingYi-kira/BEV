import torch
import torch.nn as nn
from models import get_model
from utils.data import AngleError, TransError
from models.basemodel import BaseModel
from models.model_utils import FcBlock, FeatureFusion, ECAttention, CBAM
from models.bev_projection import PolarProjectionDepth, CartesianProjection, BEV_FeatureNet

class Bevnet(BaseModel):
    default_conf = {
        "image_encoder": "???",
        "bev_net": "???",
        "pose_encoder": "???",
        "latent_dim": "???",
        "matching_dim": "???",
        "scale_range": [0, 9],
        "num_scale_bins": "???",
        "z_min": None,
        "z_max": "???",
        "x_max": "???",
        "pixel_per_meter": "???",
        "balance_factor": "???",
    }
    def _init(self, conf):
        Encoder = get_model(conf.image_encoder.get("name", "feature_extractor"))
        self.image_encoder = Encoder(conf.image_encoder)
        self.bev_feature_net = None if conf.bev_net is None else BEV_FeatureNet(conf.bev_net)
        
        ppm = conf.pixel_per_meter
        self.projection_polar = PolarProjectionDepth(
            conf.z_max,
            ppm,
            conf.scale_range,
            conf.z_min,
        )
        self.projection_bev = CartesianProjection(
            conf.z_max, conf.x_max, ppm, conf.z_min
        )
        self.scale_classifier = nn.Linear(conf.latent_dim, conf.num_scale_bins)
        
        if conf.bev_net is None:
            self.feature_projection = nn.Linear(
                conf.latent_dim, conf.matching_dim
            )
        
        self.cbam = CBAM(conf.pose_encoder.input_dim)
        self.fea_fusion = FeatureFusion(conf.pose_encoder)
        # self.attention = ECAttention()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.regressor_ori = FcBlock(conf.pose_encoder)
        self.regressor_pos = FcBlock(conf.pose_encoder)
    
    
    def ImageEnocoder(self, image):
        # Extract image features.
        level = 0
        f_image = self.image_encoder(image)["feature_maps"][level]
        
        return f_image
    
    def imageTobev(self, image, camera, device):
        pred = {}
        
        # Extract image features.
        level = 0
        f_image = self.image_encoder(image)["feature_maps"][level]

        # Scale the camera parameters.
        camera = camera.scale(1 / self.image_encoder.scales[level])
        camera = camera.to(device, non_blocking=True)

        # Estimate the monocular priors.
        pred["pixel_scales"] = scales = self.scale_classifier(f_image.moveaxis(1, -1))

        # generate the polar feature map.
        f_polar = self.projection_polar(f_image, scales, camera) # (B, C, Z, W)

        # Map to the BEV feature and validity mask.
        with torch.autocast("cuda", enabled=False):
            f_bev, valid_bev, _ = self.projection_bev(
                f_polar.float(), None, camera.float()
            )

        pred_bev = {}
        if self.conf.bev_net is None:
            # channel last -> classifier -> channel first
            f_bev = self.feature_projection(f_bev.moveaxis(1, -1)).moveaxis(-1, 1)
        else:
            pred_bev = pred["bev"] = self.bev_feature_net({"input": f_bev})
            f_bev = pred_bev["output"]

        confidence_bev = pred_bev.get("confidence")
        if confidence_bev is not None:
            f_bev = f_bev * confidence_bev.unsqueeze(1)
        f_bev = f_bev.masked_fill(~valid_bev.unsqueeze(1), 0.0)

        return f_bev

    def pose_encoder(self, fea):
        fea = self.cbam(fea)
        mid = self.fea_fusion(fea)
        # mid = self.attention(feature)
        mid = self.adaptive_pool(mid).view(mid.size(0), -1)

        pred_ori = self.regressor_ori(mid)
        pred_pos = self.regressor_pos(mid)
        
        pred_pose = torch.cat((pred_ori, pred_pos), dim=-1)

        return pred_pose

    def _forward(self, data):
        # prepare the data
        image = data["image"]
        image_v =  torch.cat((image[:, :-1], image[:, 1:]), dim=2)
        batch_size = image_v.size(0)
        seq_len = image_v.size(1)

        # (B * (S-1)), 2C, H, W)
        image_v = image_v.view(batch_size * seq_len, image_v.size(2), image_v.size(3), image_v.size(4))
        # bev_feature_f = self.imageTobev(image_v[:, 0:3, :, :], data["camera"], data["image"].device)
        # bev_feature_s = self.imageTobev(image_v[:, 3:6, :, :], data["camera"], data["image"].device)
        
        bev_feature_f = self.ImageEnocoder(image_v[:, 0:3, :, :])
        bev_feature_s = self.ImageEnocoder(image_v[:, 3:6, :, :])
        
        bev_feature = torch.cat((bev_feature_f, bev_feature_s), dim=1)
        pred_pose = self.pose_encoder(bev_feature)

        return pred_pose
    
    def loss(self, pred, data):
        gts = data["pose"].squeeze(dim=1)

        angle_loss = nn.functional.mse_loss(pred[:, :3], gts[:, :3])
        trans_loss = nn.functional.mse_loss(pred[:, 3:], gts[:, 3:])

        balance_factor = self.conf.get("balance_factor", 100)
        total_loss = (balance_factor * angle_loss + trans_loss)

        return {
            "total_loss": total_loss,
            "trans_loss": trans_loss,
            "angle_loss": angle_loss, 
        }

    def metrics(self):
        return {
            "angel_error": AngleError(),
            "trans_error": TransError(),
        }
        

        
