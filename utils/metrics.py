import sys

import torch
import torch.nn.functional as F
from ignite.engine import Engine
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce


class MeanMetric(Metric):
    def __init__(self, output_transform=lambda x: x["output"], device="cpu"):
        self._sum = None
        self._num_examples = None
        self.required_output_keys = ()
        super(MeanMetric, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._sum = torch.tensor(0, device=self._device, dtype=float)
        self._num_examples = 0
        super(MeanMetric, self).reset()

    @reinit__is_reduced
    def update(self, value):
        if torch.any(torch.isnan(torch.tensor(value))):
            return
        self._sum += value
        self._num_examples += 1

    @sync_all_reduce("_num_examples:SUM", "_sum:SUM")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('CustomAccuracy must have at least one example before it can be computed.')
        return self._sum.item() / self._num_examples

    @torch.no_grad()
    def iteration_completed(self, engine: Engine) -> None:
        output = self._output_transform(engine.state.output)
        self.update(output)



class FG_ARI(Metric):
    def __init__(self, output_transform=lambda x: x["output"], device="cpu"):
        self._sum_fg_aris = None
        self._num_examples = None
        self.required_output_keys = ()
        super(FG_ARI, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._sum_fg_aris = torch.tensor(0, device=self._device, dtype=float)
        self._num_examples = 0
        super(FG_ARI, self).reset()

    @reinit__is_reduced
    def update(self, data):
        true_masks = data["segs"]                    # fc [n, h, w]
        pred_masks = data["slot_masks"]              # n, fc, sc, h, w

        n, fc, sc, h, w = pred_masks.shape

        true_masks = [F.interpolate(tm.to(float).unsqueeze(1), (h, w), mode="nearest").squeeze(1).to(int) for tm in true_masks]

        for i in range(n):
            for f in range(fc):
                true_mask = true_masks[f][i]
                pred_mask = pred_masks[i, f]

                true_mask = true_mask.view(-1)
                pred_mask = pred_mask.view(sc, -1)

                if torch.max(true_mask) == 0:
                    continue

                foreground = true_mask > 0
                true_mask = true_mask[foreground]
                pred_mask = pred_mask[:, foreground].permute(1, 0)

                true_mask = F.one_hot(true_mask)

                # Filter out empty true groups
                not_empty = torch.any(true_mask, dim=0)
                true_mask = true_mask[:, not_empty]

                # Filter out empty predicted groups
                not_empty = torch.any(pred_mask, dim=0)
                pred_mask = pred_mask[:, not_empty]

                true_mask.unsqueeze_(0)
                pred_mask.unsqueeze_(0)

                _, n_points, n_true_groups = true_mask.shape
                n_pred_groups = pred_mask.shape[-1]
                if n_points <= n_true_groups and n_points <= n_pred_groups:
                    print("adjusted_rand_index requires n_groups < n_points.", file=sys.stderr)
                    continue

                true_group_ids = torch.argmax(true_mask, -1)
                pred_group_ids = torch.argmax(pred_mask, -1)
                true_mask_oh = true_mask.to(torch.float32)
                pred_mask_oh = F.one_hot(pred_group_ids, n_pred_groups).to(torch.float32)

                n_points = torch.sum(true_mask_oh, dim=[1, 2]).to(torch.float32)

                nij = torch.einsum('bji,bjk->bki', pred_mask_oh, true_mask_oh)
                a = torch.sum(nij, dim=1)
                b = torch.sum(nij, dim=2)

                rindex = torch.sum(nij * (nij - 1), dim=[1, 2])
                aindex = torch.sum(a * (a - 1), dim=1)
                bindex = torch.sum(b * (b - 1), dim=1)
                expected_rindex = aindex * bindex / (n_points * (n_points - 1))
                max_rindex = (aindex + bindex) / 2
                ari = (rindex - expected_rindex) / (max_rindex - expected_rindex + 0.000000000001)

                _all_equal = lambda values: torch.all(torch.eq(values, values[..., :1]), dim=-1)
                both_single_cluster = torch.logical_and(_all_equal(true_group_ids), _all_equal(pred_group_ids))

                self._sum_fg_aris += torch.where(both_single_cluster, torch.ones_like(ari), ari).squeeze()
                self._num_examples += 1

    @sync_all_reduce("_num_examples:SUM", "_sum_fg_aris:SUM")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('CustomAccuracy must have at least one example before it can be computed.')
        return self._sum_fg_aris.item() / self._num_examples

    @torch.no_grad()
    def iteration_completed(self, engine: Engine) -> None:
        output = self._output_transform(engine.state.output)
        self.update(output)
