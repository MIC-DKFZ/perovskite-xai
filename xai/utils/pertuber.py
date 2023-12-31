import numpy as np
import torch

#### From: https://github.com/CAMP-eXplain-AI/InputIBA/tree/master/input_iba/evaluation ####


class Perturber:
    def perturb(self, r: int, c: int):
        """perturb a tile or pixel"""
        raise NotImplementedError

    def get_current(self) -> np.ndarray:
        """get current img with some perturbations"""
        raise NotImplementedError

    def get_idxes(self, hmap: np.ndarray, reverse=False) -> list:
        # TODO: might not needed, we determine perturb priority outside
        #  perturber
        """return a sorted list with shape length NUM_CELLS of
        which pixel/cell index to blur first"""
        raise NotImplementedError

    def get_grid_shape(self) -> tuple:
        """return the shape of the grid, i.e. the max r, c values"""
        raise NotImplementedError


class PixelPerturber(Perturber):
    def __init__(self, inp: torch.Tensor, baseline: torch.Tensor):
        self.current = inp.clone()
        self.baseline = baseline

    def perturb(self, r: int, c: int):
        self.current[:, r, c] = self.baseline[:, r, c]

    def get_current(self) -> torch.Tensor:
        return self.current

    def get_grid_shape(self) -> tuple:
        return self.current.shape
