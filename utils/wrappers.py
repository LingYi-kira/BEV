import torch
import numpy as np
import inspect
import functools
from typing import Dict, List, NamedTuple, Tuple, Union


@torch.jit.script
def undistort_points(pts, dist):
    dist = dist.unsqueeze(-2)  # add point dimension
    ndist = dist.shape[-1]
    undist = pts
    valid = torch.ones(pts.shape[:-1], device=pts.device, dtype=torch.bool)
    if ndist > 0:
        k1, k2 = dist[..., :2].split(1, -1)
        r2 = torch.sum(pts**2, -1, keepdim=True)
        radial = k1 * r2 + k2 * r2**2
        undist = undist + pts * radial

        limited = ((k2 > 0) & ((9 * k1**2 - 20 * k2) > 0)) | ((k2 <= 0) & (k1 > 0))
        limit = torch.abs(
            torch.where(
                k2 > 0,
                (torch.sqrt(9 * k1**2 - 20 * k2) - 3 * k1) / (10 * k2),
                1 / (3 * k1),
            )
        )
        valid = valid & torch.squeeze(~limited | (r2 < limit), -1)

        if ndist > 2:
            p12 = dist[..., 2:]
            p21 = p12.flip(-1)
            uv = torch.prod(pts, -1, keepdim=True)
            undist = undist + 2 * p12 * uv + p21 * (r2 + 2 * pts**2)

    return undist, valid

def autocast(func):
    """Cast the inputs of a TensorWrapper method to PyTorch tensors
    if they are numpy arrays. Use the device and dtype of the wrapper.
    """

    @functools.wraps(func)
    def wrap(self, *args):
        device = torch.device("cpu")
        dtype = None
        if isinstance(self, TensorWrapper):
            if self._data is not None:
                device = self.device
                dtype = self.dtype
        elif not inspect.isclass(self) or not issubclass(self, TensorWrapper):
            raise ValueError(self)

        cast_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                arg = torch.from_numpy(arg)
                arg = arg.to(device=device, dtype=dtype)
            cast_args.append(arg)
        return func(self, *cast_args)

    return wrap

class TensorWrapper:
    _data = None

    @autocast
    def __init__(self, data: torch.Tensor):
        self._data = data

    @property
    def shape(self):
        return self._data.shape[:-1]

    @property
    def device(self):
        return self._data.device

    @property
    def dtype(self):
        return self._data.dtype

    def __getitem__(self, index):
        return self.__class__(self._data[index])

    def __setitem__(self, index, item):
        self._data[index] = item.data

    def to(self, *args, **kwargs):
        return self.__class__(self._data.to(*args, **kwargs))

    def cpu(self):
        return self.__class__(self._data.cpu())

    def cuda(self):
        return self.__class__(self._data.cuda())

    def pin_memory(self):
        return self.__class__(self._data.pin_memory())

    def float(self):
        return self.__class__(self._data.float())

    def double(self):
        return self.__class__(self._data.double())

    def detach(self):
        return self.__class__(self._data.detach())

    @classmethod
    def stack(cls, objects: List, dim=0, *, out=None):
        data = torch.stack([obj._data for obj in objects], dim=dim, out=out)
        return cls(data)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func is torch.stack:
            return cls.stack(*args, **kwargs)
        else:
            return NotImplemented


class Camera(TensorWrapper):
    eps = 1e-4

    def __init__(self, data: torch.Tensor):
        assert data.shape[-1] in {6, 8, 10}
        super().__init__(data)

    @classmethod
    def from_dict(cls, camera: Union[Dict, NamedTuple]):
        """Camera from a COLMAP Camera tuple or dictionary.
        We assume that the origin (0, 0) is the center of the top-left pixel.
        This is different from COLMAP.
        """
        if isinstance(camera, tuple):
            camera = camera._asdict()

        model = camera["model"]
        params = camera["params"]

        if model in ["OPENCV", "PINHOLE"]:
            (fx, fy, cx, cy), params = np.split(params, [4])
        elif model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"]:
            (f, cx, cy), params = np.split(params, [3])
            fx = fy = f
            if model == "SIMPLE_RADIAL":
                params = np.r_[params, 0.0]
        else:
            raise NotImplementedError(model)

        data = np.r_[
            camera["width"], camera["height"], fx, fy, cx - 0.5, cy - 0.5, params
        ]
        return cls(data)

    @property
    def size(self) -> torch.Tensor:
        """Size (width height) of the images, with shape (..., 2)."""
        return self._data[..., :2]

    @property
    def f(self) -> torch.Tensor:
        """Focal lengths (fx, fy) with shape (..., 2)."""
        return self._data[..., 2:4]

    @property
    def c(self) -> torch.Tensor:
        """Principal points (cx, cy) with shape (..., 2)."""
        return self._data[..., 4:6]

    @property
    def dist(self) -> torch.Tensor:
        """Distortion parameters, with shape (..., {0, 2, 4})."""
        return self._data[..., 6:]

    def scale(self, scales: Union[float, int, Tuple[Union[float, int]]]):
        """Update the camera parameters after resizing an image."""
        if isinstance(scales, (int, float)):
            scales = (scales, scales)
        s = self._data.new_tensor(scales)
        data = torch.cat(
            [self.size * s, self.f * s, (self.c + 0.5) * s - 0.5, self.dist], -1
        )
        return self.__class__(data)

    def crop(self, left_top: Tuple[float], size: Tuple[int]):
        """Update the camera parameters after cropping an image."""
        left_top = self._data.new_tensor(left_top)
        size = self._data.new_tensor(size)
        data = torch.cat([size, self.f, self.c - left_top, self.dist], -1)
        return self.__class__(data)

    @autocast
    def in_image(self, p2d: torch.Tensor):
        """Check if 2D points are within the image boundaries."""
        assert p2d.shape[-1] == 2
        # assert p2d.shape[:-2] == self.shape  # allow broadcasting
        size = self.size.unsqueeze(-2)
        valid = torch.all((p2d >= 0) & (p2d <= (size - 1)), -1)
        return valid

    @autocast
    def project(self, p3d: torch.Tensor) -> Tuple[torch.Tensor]:
        """Project 3D points into the camera plane and check for visibility."""
        z = p3d[..., -1]
        valid = z > self.eps
        z = z.clamp(min=self.eps)
        p2d = p3d[..., :-1] / z.unsqueeze(-1)
        return p2d, valid

    def J_project(self, p3d: torch.Tensor):
        x, y, z = p3d[..., 0], p3d[..., 1], p3d[..., 2]
        zero = torch.zeros_like(z)
        J = torch.stack([1 / z, zero, -x / z**2, zero, 1 / z, -y / z**2], dim=-1)
        J = J.reshape(p3d.shape[:-1] + (2, 3))
        return J  # N x 2 x 3

    @autocast
    def undistort(self, pts: torch.Tensor) -> Tuple[torch.Tensor]:
        """Undistort normalized 2D coordinates
        and check for validity of the distortion model.
        """
        assert pts.shape[-1] == 2
        # assert pts.shape[:-2] == self.shape  # allow broadcasting
        return undistort_points(pts, self.dist)

    @autocast
    def denormalize(self, p2d: torch.Tensor) -> torch.Tensor:
        """Convert normalized 2D coordinates into pixel coordinates."""
        return p2d * self.f.unsqueeze(-2) + self.c.unsqueeze(-2)

    @autocast
    def normalize(self, p2d: torch.Tensor) -> torch.Tensor:
        """Convert pixel coordinates into normalized 2D coordinates."""
        return (p2d - self.c.unsqueeze(-2)) / self.f.unsqueeze(-2)

    def J_denormalize(self):
        return torch.diag_embed(self.f).unsqueeze(-3)  # 1 x 2 x 2

    @autocast
    def world2image(self, p3d: torch.Tensor) -> Tuple[torch.Tensor]:
        """Transform 3D points into 2D pixel coordinates."""
        p2d, visible = self.project(p3d)
        p2d, mask = self.undistort(p2d)
        p2d = self.denormalize(p2d)
        valid = visible & mask & self.in_image(p2d)
        return p2d, valid

    def J_world2image(self, p3d: torch.Tensor):
        p2d_dist, valid = self.project(p3d)
        J = self.J_denormalize() @ self.J_undistort(p2d_dist) @ self.J_project(p3d)
        return J, valid

    def __repr__(self):
        return f"Camera {self.shape} {self.dtype} {self.device}"