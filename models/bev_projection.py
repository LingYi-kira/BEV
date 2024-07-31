import torch
import torch.nn as nn
from torch.nn.functional import grid_sample
from torchvision.models.resnet import Bottleneck

from models.basemodel import BaseModel
from models.model_utils import from_homogeneous, make_grid, checkpointed



class PolarProjectionDepth(torch.nn.Module):
    def __init__(self, z_max, ppm, scale_range, z_min=None):
        super().__init__()
        self.z_max = z_max
        self.Δ = Δ = 1 / ppm
        self.z_min = z_min = Δ if z_min is None else z_min
        self.scale_range = scale_range
        # 相机前采样的D个深度平面
        z_steps = torch.arange(z_min, z_max + Δ, Δ)
        self.register_buffer("depth_steps", z_steps, persistent=False)

    def sample_depth_scores(self, pixel_scales, camera):
        # 计算尺度步长
        scale_steps = camera.f[..., None, 1] / self.depth_steps.flip(-1)
        log_scale_steps = torch.log2(scale_steps)
        # 归一化尺度步长
        scale_min, scale_max = self.scale_range
        log_scale_norm = (log_scale_steps - scale_min) / (scale_max - scale_min)
        log_scale_norm = log_scale_norm * 2 - 1  # in [-1, 1]
        
        # 计算深度分数  .flatten(1, 2) 将第一维度和第二维度数据展开     .unsqueeze(-1) 数据扩充，在最后一维新增一个维度。
        values = pixel_scales.flatten(1, 2).unsqueeze(-1)
        indices = log_scale_norm.unsqueeze(-1)
        # 将全零张量和indices 沿着新的维度（最后一维）进行堆叠。
        indices = torch.stack([torch.zeros_like(indices), indices], -1)
        # 用于根据给定的插值索引 indices 对 values 进行插值计算。
        depth_scores = grid_sample(values, indices, align_corners=True)
        depth_scores = depth_scores.reshape(
            pixel_scales.shape[:-1] + (len(self.depth_steps),)
        )
        return depth_scores

    def forward(
        self,
        image,
        pixel_scales,
        camera,
        return_total_score=False,
    ):
        depth_scores = self.sample_depth_scores(pixel_scales, camera)
        depth_prob = torch.softmax(depth_scores, dim=1)  # (B, C, Z, W)
        # 将 image 在相应的高度和宽度上与 depth_prob 进行乘积和求和，形成新的张量维度。
        image_polar = torch.einsum("...dhw,...hwz->...dzw", image, depth_prob)
        if return_total_score:
            cell_score = torch.logsumexp(depth_scores, dim=1, keepdim=True)
            return image_polar, cell_score.squeeze(1)
        return image_polar
    

class CartesianProjection(torch.nn.Module):
    def __init__(self, z_max, x_max, ppm, z_min=None):
        super().__init__()
        self.z_max = z_max
        self.x_max = x_max
        self.Δ = Δ = 1 / ppm
        self.z_min = z_min = Δ if z_min is None else z_min

        # 生成 XZ 网格
        grid_xz = make_grid(
            x_max * 2 + Δ, z_max, step_y=Δ, step_x=Δ, orig_y=Δ, orig_x=-x_max, y_up=True
        ) #(x_max * 2 + Δ)
        self.register_buffer("grid_xz", grid_xz, persistent=False)

    def grid_to_polar(self, camera):
        # 相机焦距和光心
        f, c = camera.f[..., 0][..., None, None], camera.c[..., 0][..., None, None]
        # 将齐次坐标转换为常规坐标，计算每个点在图像平面上的横坐标 u = grid_xz * f + c
        u = from_homogeneous(self.grid_xz).squeeze(-1) * f + c
        # 计算每个点在深度轴上的索引
        z_idx = (self.grid_xz[..., 1] - self.z_min) / self.Δ  # convert z value to index
        # 扩展维度
        z_idx = z_idx[None].expand_as(u)
        grid_polar = torch.stack([u, z_idx], -1)

        return grid_polar

    # 从极坐标图像中采样数据，并生成以笛卡尔坐标表示的BEV图像。
    # 同时，还会返回一个有效性掩码，指示哪些像素是有效的。
    def sample_from_polar(self, image_polar, valid_polar, grid_uz):
        # 获取极坐标图像的高度和宽度，并翻转顺序使其与 grid_uz 对应，最后将这些尺寸转换为与 grid_uz 相同设备和数据类型的张量
        size = grid_uz.new_tensor(image_polar.shape[-2:][::-1])
        grid_uz_norm = (grid_uz + 0.5) / size * 2 - 1
        # 翻转 Y 轴方向
        grid_uz_norm = grid_uz_norm * grid_uz.new_tensor([1, -1])  
        # 使用规范化后的采样点从 image_polar 中采样数据，生成 BEV 图像
        image_bev = grid_sample(image_polar, grid_uz_norm, align_corners=False)

        # 创建有效掩码
        if valid_polar is None:
            valid = torch.ones_like(image_polar[..., :1, :, :])
        else:
            valid = valid_polar.to(image_polar)[:, None]
        valid = grid_sample(valid, grid_uz_norm, align_corners=False)
        valid = valid.squeeze(1) > (1 - 1e-4)

        return image_bev, valid

    def forward(self, image_polar, valid_polar, camear):
        # 将网格坐标转换为极坐标
        grid_uz = self.grid_to_polar(camear)
        # 输入（极坐标特征图像, None, 极坐标网格），输出BEV图像特征。
        image, valid = self.sample_from_polar(image_polar, valid_polar, grid_uz)
        
        return image, valid, grid_uz


class AdaptationBlock(nn.Sequential):
    def __init__(self, inp, out):
        conv = nn.Conv2d(inp, out, kernel_size=1, padding=0, bias=True)
        super().__init__(conv)

class BEV_FeatureNet(BaseModel):
    default_conf = {
        "pretrained": True,
        "num_blocks": "???",
        "latent_dim": "???",
        "input_dim": "${.latent_dim}",
        "output_dim": "${.latent_dim}",
        "confidence": False,
        "norm_layer": "nn.BatchNorm2d",  # normalization ind decoder blocks
        "checkpointed": False,  # whether to use gradient checkpointing
        "padding": "zeros",
    }

    def _init(self, conf):
        blocks = []
        # 实现梯度检查点保存
        Block = checkpointed(Bottleneck, do=conf.checkpointed)
        for i in range(conf.num_blocks):
            dim = conf.input_dim if i == 0 else conf.latent_dim
            blocks.append(
                Block(
                    dim,
                    conf.latent_dim // Bottleneck.expansion,   # Bottleneck.expansion = 4 
                    norm_layer=eval(conf.norm_layer),  # 使用eval函数将conf.norm_layer字符串转为实际的规范化层类
                )
            )
        self.blocks = nn.Sequential(*blocks)
        # self.output_layer = AdaptationBlock(conf.latent_dim, conf.output_dim)
        if conf.confidence:
            self.confidence_layer = AdaptationBlock(conf.latent_dim, 1)

        def update_padding(module):
            if isinstance(module, nn.Conv2d):
                module.padding_mode = conf.padding

        if conf.padding != "zeros":
            self.bocks.apply(update_padding)

    def _forward(self, data):
        features = self.blocks(data["input"])
        pred = {
            "output": features,
        }
        # 如果配置中启用了置信度输出，则通过confidence_layer计算置信度，并对结果进行sigmoid激活
        if self.conf.confidence:
            pred["confidence"] = self.confidence_layer(features).squeeze(1).sigmoid()
        return pred

    def loss(self, pred, data):
        raise NotImplementedError

    def metrics(self, pred, data):
        raise NotImplementedError