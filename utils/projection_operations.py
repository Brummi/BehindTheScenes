import torch


def distance_to_z(depths: torch.Tensor, projs: torch.Tensor):
    n, nv, h, w = depths.shape
    device = depths.device

    inv_K = torch.inverse(projs)

    grid_x = torch.linspace(-1, 1, w, device=device).view(1, 1, 1, -1).expand(-1, -1, h, -1)
    grid_y = torch.linspace(-1, 1, h, device=device).view(1, 1, -1, 1).expand(-1, -1, -1, w)
    img_points = torch.stack((grid_x, grid_y, torch.ones_like(grid_x)), dim=2).expand(n, nv, -1, -1, -1)
    cam_points = (inv_K @ img_points.view(n, nv, 3, -1)).view(n, nv, 3, h, w)
    factors = cam_points[:, :, 2, :, :] / torch.norm(cam_points, dim=2)

    return depths * factors
