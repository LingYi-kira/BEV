import cv2
import math
import torch
import collections
import torchmetrics
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as tvf
from torchmetrics.utilities.data import dim_zero_cat


_EPS = np.finfo(float).eps * 4.0

def euler_from_matrix(matrix):
    '''
    Extract the eular angle from a rotation matrix
    '''
    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    cy = math.sqrt(M[0, 0] * M[0, 0] + M[1, 0] * M[1, 0])
    ay = math.atan2(-M[2, 0], cy)
    if ay < -math.pi / 2 + _EPS and ay > -math.pi / 2 - _EPS:  # pitch = -90 deg
        ax = 0
        az = math.atan2(-M[1, 2], -M[0, 2])
    elif ay < math.pi / 2 + _EPS and ay > math.pi / 2 - _EPS:
        ax = 0
        az = math.atan2(M[1, 2], M[0, 2])
    else:
        ax = math.atan2(M[2, 1], M[2, 2])
        az = math.atan2(M[1, 0], M[0, 0])
    return np.array([ax, ay, az])

def get_relative_pose(Rt1, Rt2):
    '''
    Calculate the relative 4x4 pose matrix between two pose matrices
    '''
    Rt1_inv = np.linalg.inv(Rt1)
    Rt_rel = Rt1_inv @ Rt2
    return Rt_rel

def get_relative_pose_6DoF(Rt1, Rt2):
    '''
    Calculate the relative rotation and translation from two consecutive pose matrices 
    '''
    
    # Calculate the relative transformation Rt_rel
    Rt_rel = get_relative_pose(Rt1, Rt2)

    R_rel = Rt_rel[:3, :3]
    t_rel = Rt_rel[:3, 3]

    # Extract the Eular angle from the relative rotation matrix
    x, y, z = euler_from_matrix(R_rel)
    theta = [x, y, z]

    pose_rel = np.concatenate((theta, t_rel))
    return pose_rel

def read_pose(line):
    '''
    Reading 4x4 pose matrix from .txt files
    input: a line of 12 parameters
    output: 4x4 numpy matrix
    '''
    values= np.reshape(np.array([float(value) for value in line.split(' ')]), (3, 4))
    Rt = np.concatenate((values, np.array([[0, 0, 0, 1]])), 0)
    return Rt.astype(np.float32)

def read_pose_from_text(path):
    with open(path) as f:
        lines = [line.split('\n')[0] for line in f.readlines()]
        poses_rel, poses_abs = [], []
        values_p = read_pose(lines[0])
        poses_abs.append(values_p)
        for i in range(1, len(lines)):
            values = read_pose(lines[i])
            poses_rel.append(get_relative_pose_6DoF(values_p, values)) 
            values_p = values.copy()
            poses_abs.append(values) 
        poses_abs = np.array(poses_abs)
        poses_rel = np.array(poses_rel)
    
    return poses_abs, poses_rel

def rotationError(Rt1, Rt2):
    '''
    Calculate the rotation difference between two pose matrices
    '''
    pose_error = get_relative_pose(Rt1, Rt2)
    a = pose_error[0, 0]
    b = pose_error[1, 1]
    c = pose_error[2, 2]
    d = 0.5 * (a + b + c - 1.0)
    return np.arccos(max(min(d, 1.0), -1.0))

def read_calib(path):
    with open(path) as f:
        lines = [line.split('\n')[0] for line in f.readlines()]
        p2_values = lines[1].split()[1:]
        p2_values = list(map(float, p2_values))
        params_K = np.array([p2_values[0], p2_values[5], p2_values[2], p2_values[6]])

    return {'model': 'PINHOLE', 
             'width': 1242, 
             'height': 375, 
             'params': params_K,
    }

class ToTensor(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, images, imus, gts, camera, seed):
        tensors = []
        for im in images:
            image = (
                torch.from_numpy(np.ascontiguousarray(im))
                .permute(2, 0, 1)
                .float()
                .div_(255)
            )
            # data enhancement
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                image = self.func(image)
            tensors.append(image)

        tensors = torch.stack(tensors, 0)
        gts = torch.tensor(gts).float()
        return tensors, imus, gts, camera
    
    def __repr__(self):
        return self.__class__.__name__ + '()'


class Resize(object):
    def __init__(self, size=(256, 512)):
        if isinstance(size, (collections.abc.Sequence, np.ndarray)):
            h_new, w_new  = (int(x) for x in size)
        elif isinstance(size, int):
            w_new = h_new = size
        else:
            raise ValueError(f"Incorrect new size: {size}")
        self.h_new = h_new
        self.w_new = w_new
        

    def __call__(self, images, imus, gts, camera, seed):
        tensors = []
        *_, h, w = images[0].shape
        scale = (self.w_new / w, self.h_new / h)
        for im in images:
            if isinstance(im, np.ndarray):
                im = torch.from_numpy(im).permute(2, 0, 1).float()
            elif isinstance(im, torch.Tensor):
                im = im.float()
            # Resize the tensor
            mode = tvf.InterpolationMode.BILINEAR
            im = tvf.resize(im, (self.h_new, self.w_new), interpolation=mode, antialias=True)
            im.clip_(0, 1)
            tensors.append(im)

        camera = camera.scale(scale)
        tensors = torch.stack(tensors, 0)
        
        return tensors, imus, gts, camera
        
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'img_h: {}, '.format(self.size[0])
        format_string += 'img_w: {})'.format(self.size[1])
        return format_string

def read_image(path, grayscale=False):
    # 读取灰色/彩色图像
    if grayscale:
        mode = cv2.IMREAD_GRAYSCALE
    else:
        mode = cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    # 检查是否读取成功
    if image is None:
        raise ValueError(f"Cannot read image {path}.")
    # OpenCV 默认以 BGR 格式读取彩色图像
    if not grayscale and len(image.shape) == 3:
        image = np.ascontiguousarray(image[:, :, ::-1])  # BGR to RGB
    return image



class MeanMetricWithRecall(torchmetrics.Metric):
    full_state_update = True

    def __init__(self):
        super().__init__()
        self.add_state("value", default=[], dist_reduce_fx="cat")

    def compute(self):
        return dim_zero_cat(self.value).mean(0)

    def get_errors(self):
        return dim_zero_cat(self.value)


def euler_to_quaternion(data):
    """
    Convert Euler Angles to Quaternion.
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.
    :return: A tensor of shape (4,) representing the quaternion [w, x, y, z].
    """
    roll, pitch, yaw = torch.split(data, 1, dim=1)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return torch.stack([qw, qx, qy, qz])

def angle_error(pred, real):
    q_pred = euler_to_quaternion(pred)
    q_real = euler_to_quaternion(real)

	#归一化
    q1 = q_pred / torch.norm(q_pred, p=2)  
    q2 = q_real / torch.norm(q_real, p=2)

    #计算角度
    dot_product = torch.sum(q1 * q2)
    d = torch.clamp(dot_product.abs(), min=-1.0, max=1.0)
    theta = 2 * torch.acos(d) * 180 / torch.tensor(torch.pi)

    return theta

class AngleError(MeanMetricWithRecall):
    def __init__(self):
        super().__init__()

    def update(self, pred, data):
        value = angle_error(pred[:, 0:3], data["pose"].squeeze(dim=1)[:, 0:3])
        if value.numel():
            self.value.append(value)

def trans_error(pred, real):
    error = torch.norm(real - pred, p=2)

    return error

class TransError(MeanMetricWithRecall):
    def __init__(self):
        super().__init__()

    def update(self, pred, data):
        value = trans_error(pred[:, 3:6], data["pose"].squeeze(dim=1)[:, 3:6])
        if value.numel():
            self.value.append(value)