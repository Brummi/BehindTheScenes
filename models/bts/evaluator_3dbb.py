import math

import torch
from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Engine
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from datasets.data_util import make_test_dataset
from datasets.kitti_360.labels import id2label
from models.common.render import NeRFRenderer
from models.bts.model.image_processor import make_image_processor, RGBProcessor
from models.bts.model.loss import ReconstructionLoss
from models.bts.model.models_bts import BTSNet
from models.bts.model.ray_sampler import ImageRaySampler, PatchRaySampler, RandomRaySampler
from utils.base_evaluator import base_evaluation
from utils.metrics import MeanMetric
from utils.projection_operations import distance_to_z

IDX = 0


EPS = 1e-4

bad = []


def verts_to_cam(bbox, pose):
    verts = bbox["vertices"][0].to(torch.float32)
    verts = pose[:3, :3] @ verts.T + pose[:3, 3, None]
    bbox["vertices"] = verts.T
    bbox["faces"] = bbox["faces"][0].to(int)
    return bbox


def bbox_in_frustum(bbox, projs, max_d, reducer=torch.any):
    verts = bbox["vertices"]
    verts = (projs @ verts.T).T
    verts[:, :2] /= verts[:, 2:3]
    valid = ((verts[:, 0] >= -1) & (verts[:, 0] <= 1)) & ((verts[:, 1] >= -1) & (verts[:, 1] <= 1)) & ((verts[:, 2] > 0) & (verts[:, 2] <= max_d))
    valid = reducer(valid, dim=-1)
    return valid


def compute_bounds(bbox):
    vertices = bbox["vertices"]
    faces = bbox["faces"]

    # (n, 3)
    face_normals = torch.cross(torch.index_select(vertices, index=faces[:, 1], dim=0) - torch.index_select(vertices, index=faces[:, 0], dim=0),
                            torch.index_select(vertices, index=faces[:, 2], dim=0) - torch.index_select(vertices, index=faces[:, 0], dim=0))
    face_normals /= torch.norm(face_normals, dim=-1, keepdim=True)

    projections = face_normals @ vertices.T
    projections_min = projections.min(dim=-1, keepdim=True)[0]
    projections_max = projections.max(dim=-1, keepdim=True)[0]
    face_normals_bounds = torch.cat((face_normals, projections_min, projections_max), dim=-1)
    return face_normals_bounds


def in_bbox(pts, fnbs):
    # pts (n, 3)
    # fnbs (m, 5)

    # (m, n)
    projections = fnbs[:, :3] @ pts.T
    # (m, n)
    is_in = (fnbs[:, 3:4] - EPS <= projections) & (projections <= fnbs[:, 4:5] + EPS)
    is_in = torch.all(is_in, dim=0)

    # (n, )
    return is_in


def bbox_intercept(d, fnbs):
    # Everything is based on computation in camera space (origin of rays is (0, 0, 0))
    # d (n, 3)
    # fnbs (m, 5)

    n, _ = d.shape
    m, _ = fnbs.shape

    # (m, n)
    denom = fnbs[:, :3] @ d.T
    i1 = (fnbs[:, 3:4] / denom).T.unsqueeze(-1) * d.unsqueeze(1)
    i2 = (fnbs[:, 4:5] / denom).T.unsqueeze(-1) * d.unsqueeze(1)

    # (n, 2m, 3)
    pts = torch.cat((i1, i2), dim=1).view(-1, 3)
    is_in = in_bbox(pts, fnbs) & (pts[:, 2] > 0)
    pts[~is_in, :] = float("inf")
    pts = pts.view(n, 2*m, 3)
    intercept_i = torch.argmin(pts[:, :, 2], dim=1)
    pts_intercept = pts[torch.arange(len(intercept_i)), intercept_i, :]

    # (n, 3)
    return pts_intercept


def bbox_intercept_labeled(d, l, fnbs, box_label):
    # Everything is based on computation in camera space (origin of rays is (0, 0, 0))
    # d (n, 3)
    # l (n, )
    # fnbs (m, 5)
    # box_labels (1, )

    n, _ = d.shape
    m, _ = fnbs.shape

    # (m, n)
    denom = fnbs[:, :3] @ d.T
    i1 = (fnbs[:, 3:4] / denom).T.unsqueeze(-1) * d.unsqueeze(1)
    i2 = (fnbs[:, 4:5] / denom).T.unsqueeze(-1) * d.unsqueeze(1)

    # (n, 2m, 3)
    pts = torch.cat((i1, i2), dim=1).view(-1, 3)
    is_in = in_bbox(pts, fnbs) & (pts[:, 2] > 0)
    is_label = (l.view(n, 1) == box_label).expand(n, 2*m).reshape(-1)
    is_valid = is_in & is_label
    pts[~is_valid, :] = float("inf")
    pts = pts.view(n, 2*m, 3)
    intercept_i = torch.argmin(pts[:, :, 2], dim=1)
    pts_intercept = pts[torch.arange(len(intercept_i)), intercept_i, :]

    # (n, 3)
    return pts_intercept


def get_pts(x_range, y_range, z_range, ppm, ppm_y):
    x_res = abs(int((x_range[1] - x_range[0]) * ppm))
    y_res = abs(int((y_range[1] - y_range[0]) * ppm_y))
    z_res = abs(int((z_range[1] - z_range[0]) * ppm))
    x = torch.linspace(x_range[0], x_range[1], x_res).view(1, 1, x_res).expand(y_res, z_res, -1)
    z = torch.linspace(z_range[0], z_range[1], z_res).view(1, z_res, 1).expand(y_res, -1, x_res)
    y = torch.linspace(y_range[0], y_range[1], y_res).view(y_res, 1, 1).expand(-1, z_res, x_res)
    xyz = torch.stack((x, y, z), dim=-1)

    # The KITTI 360 cameras have a 5 degrees negative inclination. We need to account for that tan(5°) = 0.0874886635
    xyz[:, :, :, 1] -= xyz[:, :, :, 2] * 0.0874886635

    return xyz, (x_res, y_res, z_res)


def project_into_cam(pts, proj):
    cam_pts = (proj @ pts.T).T
    cam_pts[:, :2] /= cam_pts[:, 2:3]
    dist = cam_pts[:, 2]
    return cam_pts, dist


def visualize(pts, xd, yd, zd):
    pts = pts.reshape(yd, zd, xd).cpu().numpy()

    rows = math.ceil(yd / 2)
    fig, axs = plt.subplots(rows, 2)

    for y in range(yd):
        r = y // 2
        c = y % 2

        axs[r][c].imshow(pts[y], interpolation="none")

    plt.show()


class BTSWrapper(nn.Module):
    def __init__(self, renderer, config, ) -> None:
        super().__init__()

        self.renderer = renderer

        self.z_near = config["z_near"]
        self.z_far = config["z_far"]
        self.query_batch_size = config.get("query_batch_size", 50000)
        self.occ_threshold = 0.5

        self.x_range = (-4, 4)
        self.y_range = (0, 1)
        # self.z_range = (19, 3)
        self.z_range = (20, 3)
        self.ppm = 5
        self.ppm_y = 4

        self.sampler = ImageRaySampler(self.z_near, self.z_far, channels=1)


    @staticmethod
    def get_loss_metric_names():
        return ["loss", "loss_l2", "loss_mask", "loss_temporal"]

    def forward(self, data):
        data = dict(data)
        images = torch.stack(data["imgs"], dim=1)                           # n, v, c, h, w
        poses = torch.stack(data["poses"], dim=1)                           # n, v, 4, 4 w2c
        projs = torch.stack(data["projs"], dim=1)                           # n, v, 4, 4 (-1, 1)
        bboxes = data["3d_bboxes"][0]
        seg = data["segs"][0]

        bboxes = [bbox for bbox in bboxes if id2label[bbox["semanticId"].item()].category != "flat"]

        n, v, c, h, w = images.shape
        device = images.device

        ph = h // 2
        pw = w // 2

        self.sampler.height = ph
        self.sampler.width = pw

        to_keyframe = torch.inverse(poses[:, :1, :, :])
        
        # Everything is implemented for batch size 1
        verts_to_cam_ = lambda bbox: verts_to_cam(bbox, to_keyframe[0, 0])
        bbox_in_frustum_ = lambda bbox: bbox_in_frustum(bbox, projs[0, 0], self.z_range[0], reducer=torch.any)

        bboxes = list(map(verts_to_cam_, bboxes))
        bboxes = list(filter(bbox_in_frustum_, bboxes))

        is_bad = len([bbox for bbox in bboxes if id2label[bbox["semanticId"].item()].category in ["human", "vehicle"]]) == 0
        if is_bad:
            bad.append(globals()["IDX"])
            print("bad: ", bad)

        fnbs = list(map(compute_bounds, bboxes))
        labels = torch.tensor([bbox["semanticId"] for bbox in bboxes], device=fnbs[0].device)

        poses = to_keyframe @ poses

        rays, gt_label = self.sampler.sample(F.interpolate(seg.unsqueeze(0), (self.sampler.height, self.sampler.width), mode="nearest"), poses[:, :1, :, :], projs[:, :1, :, :], )

        # (n, 3)
        dirs = rays[0, :, 3:6].view(-1, 3)

        bbox_intercept_labeled_ = lambda fnb_box_label: bbox_intercept_labeled(dirs, gt_label, fnb_box_label[0], fnb_box_label[1])
        pts_intercept = torch.stack(list(map(bbox_intercept_labeled_, zip(fnbs, labels))), dim=1)
        intercept_i = torch.argmin(pts_intercept[:, :, 2], dim=1)
        pts_intercept = pts_intercept[torch.arange(len(intercept_i)), intercept_i, :]

        pseudo_depth = pts_intercept[:, 2].view(ph, pw)

        ids_encoder = [0]
        self.renderer.net.compute_grid_transforms(projs[:, ids_encoder], poses[:, ids_encoder])
        self.renderer.net.encode(images, projs, poses, ids_encoder=ids_encoder, ids_render=ids_encoder, images_alt=images.mean(2, keepdim=True) * .5 + .5)
        self.renderer.net.set_scale(0)
        render_dict = self.renderer(rays, want_weights=True, want_alphas=True)
        if "fine" not in render_dict:
            render_dict["fine"] = dict(render_dict["coarse"])
        render_dict = self.sampler.reconstruct(render_dict)
        pred_depth = distance_to_z(render_dict["coarse"]["depth"], projs[:1, :1])

        # Get pts
        q_pts, (xd, yd, zd) = get_pts(self.x_range, self.y_range, self.z_range, self.ppm, self.ppm_y)
        q_pts = q_pts.to(pseudo_depth.device).view(-1, 3)

        # is visible? Check whether point is closer than the computed pseudo depth
        cam_pts, dists = project_into_cam(q_pts, projs[0, 0])
        gt_dist = F.grid_sample(pseudo_depth.view(1, 1, ph, pw), cam_pts[:, :2].view(1, 1, -1, 2), mode="nearest", padding_mode="border", align_corners=True).view(-1)
        pred_dist = F.grid_sample(pred_depth.view(1, 1, ph, pw), cam_pts[:, :2].view(1, 1, -1, 2), mode="nearest", padding_mode="border", align_corners=True).view(-1)
        is_visible = (dists <= gt_dist) | (dists <= pred_dist)

        # is occupied?
        is_occupied = torch.zeros_like(q_pts[:, 0], dtype=torch.bool)
        for fnb, bbox in zip(fnbs, bboxes):
            id = bbox["semanticId"]
            label = id2label[id.item()]
            # The flat category contains street, sidewalk, etc.
            # It seems that these bounding boxes are too high and would disturb the final result.
            # We only check points above the street anyways.
            if label.category == "flat":
                continue
            is_occupied |= in_bbox(q_pts, fnb)
        # Only not visible points can be occupied
        is_occupied &= ~is_visible

        # Query the density of the query points from the density field
        densities = []
        for i_from in range(0, len(q_pts), self.query_batch_size):
            i_to = min(i_from + self.query_batch_size, len(q_pts))
            q_pts_ = q_pts[i_from:i_to]
            _, _, densities_ = self.renderer.net(q_pts_.unsqueeze(0), only_density=True)
            densities.append(densities_.squeeze(0))
        densities = torch.cat(densities, dim=0).squeeze()

        is_occupied_pred = densities > self.occ_threshold

        is_occupied_acc = (is_occupied_pred == is_occupied).float().mean().item()
        is_occupied_prec = is_occupied[is_occupied_pred].float().mean().item()
        is_occupied_rec = is_occupied_pred[is_occupied].float().mean().item()

        not_occupied_not_visible_ratio = ((~is_occupied) & (~is_visible)).float().mean().item()

        total_no_nv = ((~is_occupied) & (~is_visible)).float().sum().item()

        no_nv_acc = (is_occupied_pred == is_occupied)[(~is_visible)].float().mean().item()
        no_nv_prec = (~is_occupied)[(~is_occupied_pred) & (~is_visible)].float().mean()
        no_nv_rec = (~is_occupied_pred)[(~is_occupied) & (~is_visible)].float().mean()
        total_no_nop_nv = ((~is_occupied) & (~is_occupied_pred))[(~is_visible) & (~is_occupied)].float().sum()

        data["o_acc"] = is_occupied_acc
        data["o_rec"] = is_occupied_rec
        data["o_prec"] = is_occupied_prec
        data["no_nv_acc"] = no_nv_acc
        data["no_nv_rec"] = no_nv_rec
        data["no_nv_prec"] = no_nv_prec
        data["no_nv_r"] = not_occupied_not_visible_ratio
        data["t_no_nv"] = total_no_nv
        data["t_no_nop_nv"] = total_no_nop_nv

        data["z_near"] = torch.tensor(self.z_near, device=images.device)
        data["z_far"] = torch.tensor(self.z_far, device=images.device)

        globals()["IDX"] += 1

        return data


def evaluation(local_rank, config):
    return base_evaluation(local_rank, config, get_dataflow, initialize, get_metrics)


def get_dataflow(config):
    test_dataset = make_test_dataset(config["data"])
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=config["num_workers"], shuffle=False, drop_last=False)

    return test_loader


def get_metrics(config, device):
    names = ["o_acc", "o_prec", "o_rec", "no_nv_acc", "no_nv_prec", "no_nv_rec", "t_no_nv", "t_no_nop_nv"]
    metrics = {name: MeanMetric((lambda n: lambda x: x["output"][n])(name), device) for name in names}
    return metrics


def initialize(config: dict, logger=None):
    arch = config["model_conf"].get("arch", "BTSNet")
    net = globals()[arch](config["model_conf"])
    renderer = NeRFRenderer.from_conf(config["renderer"])
    renderer = renderer.bind_parallel(net, gpus=None).eval()

    model = BTSWrapper(
        renderer,
        config["model_conf"]
    )

    return model


def visualize(engine: Engine, logger: TensorboardLogger, step: int, tag: str):
    pass