import numpy as np
import torch

from scipy.ndimage import gaussian_filter1d
from abc import ABCMeta, abstractmethod
from scipy.integrate import trapezoid
from torchvision.transforms import GaussianBlur

from xai.utils.pertuber import PixelPerturber

#### From: https://github.com/CAMP-eXplain-AI/InputIBA/tree/master/input_iba/evaluation ####


class BaseEvaluation(metaclass=ABCMeta):
    """
    Base class for all evaluation methods
    get attribution map and img as input, returns a dictionary contains
    evaluation result
    """

    @abstractmethod
    def evaluate(self, heatmap, *args, **kwargs) -> dict:
        raise NotImplementedError


class VisionSensitivityN(BaseEvaluation):
    def __init__(self, classifier, input_size, n, num_masks=100):
        self.classifier = classifier
        self.n = n
        self.device = next(self.classifier.parameters()).device
        self.indices, self.masks = self._generate_random_masks(
            num_masks, input_size, device=self.device
        )

    def evaluate(  # noqa
        self,
        heatmap: torch.Tensor,
        input_tensor: torch.Tensor,
        target: int,
        calculate_corr=False,
    ) -> dict:
        pertubated_inputs = []
        sum_attributions = []
        for mask in self.masks:
            # perturb is done by interpolation
            pertubated_inputs.append(input_tensor * (1 - mask))
            sum_attributions.append((heatmap * mask).sum())
        sum_attributions = torch.stack(sum_attributions)
        input_inputs = pertubated_inputs + [input_tensor]
        with torch.no_grad():
            input_inputs = torch.stack(input_inputs).to(self.device)
            output = self.classifier(input_inputs)
        output_pertubated = output[:-1]
        output_clean = output[-1:]

        diff = output_clean[:, target] - output_pertubated[:, target]
        score_diffs = diff.cpu().numpy()
        sum_attributions = sum_attributions.detach().cpu().numpy()

        # calculate correlation for single image if requested
        corrcoef = None
        if calculate_corr:
            corrcoef = np.corrcoef(sum_attributions.flatten(), score_diffs.flatten())
        return {
            "correlation": corrcoef,
            "score_diffs": score_diffs,
            "sum_attributions": sum_attributions,
        }

    def _generate_random_masks(self, num_masks, input_size, device="cuda:0"):
        """
        generate random masks with n pixel set to zero
        Args:
            num_masks: number of masks
            n: number of perturbed pixels
        Returns:
            masks
        """
        indices = []
        masks = []
        h, w = input_size
        for _ in range(num_masks):
            idxs = np.unravel_index(
                np.random.choice(h * w, self.n, replace=False), (h, w)
            )
            indices.append(idxs)
            mask = np.zeros((h, w))
            mask[idxs] = 1
            masks.append(torch.from_numpy(mask).to(torch.float32).to(device))
        return indices, masks


class VisionInsertionDeletion(BaseEvaluation):
    def __init__(
        self, classifier, baseline, pixel_batch_size=10, kernel_size=4, sigma=5.0
    ):
        self.model = classifier
        self.model.eval()
        self.pixel_batch_size = pixel_batch_size
        self.sigma = sigma
        self.baseline = baseline.unsqueeze(0)

    @torch.no_grad()
    def evaluate(self, heatmap, input_tensor, target):  # noqa
        """# TODO to add docs
        Args:
            heatmap (Tensor): heatmap with shape (H, W) or (3, H, W).
            input_tensor (Tensor): image with shape (3, H, W).
            target (int): class index of the image.
        Returns:
            dict[str, Union[Tensor, np.array, float]]: a dictionary
                containing following fields:
                - del_scores: ndarray,
                - ins_scores:
                - del_input:
                - ins_input:
                - ins_auc:
                - del_auc:
        """

        # sort pixel in attribution
        num_pixels = torch.numel(heatmap)
        _, indices = torch.topk(heatmap.flatten(), num_pixels)
        indices = np.unravel_index(indices.cpu().numpy(), heatmap.size())

        # apply deletion game
        deletion_perturber = PixelPerturber(input_tensor, self.baseline)
        deletion_scores = self._procedure_perturb(
            deletion_perturber, num_pixels, indices, target
        )

        # apply insertion game
        blurred_input = torch.Tensor(gaussian_filter1d(input_tensor, self.sigma))
        insertion_perturber = PixelPerturber(blurred_input, input_tensor)
        insertion_scores = self._procedure_perturb(
            insertion_perturber, num_pixels, indices, target
        )

        # calculate ABC
        steps = len(insertion_scores)

        line = np.linspace(insertion_scores[0], insertion_scores[-1], steps)
        ins_line = np.where(insertion_scores < line, line, insertion_scores)
        insertion_auc = trapezoid(ins_line, dx=1.0 / len(ins_line))
        aul = trapezoid(line, dx=1.0 / steps)

        insertion_abc = insertion_auc - aul

        line = np.linspace(deletion_scores[0], deletion_scores[-1], steps)
        del_line = np.where(deletion_scores > line, line, deletion_scores)
        deletion_auc = trapezoid(del_line, dx=1.0 / len(del_line))
        aul = trapezoid(line, dx=1.0 / steps)

        deletion_abc = aul - deletion_auc

        # deletion_input and insertion_input are final results, they are
        # only used for debug purpose
        # TODO check if it is necessary to convert the Tensors to np.ndarray
        return {
            "del_scores": deletion_scores,
            "ins_scores": insertion_scores,
            "del_input": deletion_perturber.get_current(),
            "ins_input": insertion_perturber.get_current(),
            "ins_abc": insertion_abc,
            "del_abc": deletion_abc,
        }

    def _procedure_perturb(self, perturber, num_pixels, indices, target):
        """# TODO to add docs
        Args:
            perturber (PixelPerturber):
            num_pixels (int):
            indices (tuple):
            target (int):
        Returns:
            np.ndarray:
        """
        scores_after_perturb = []
        replaced_pixels = 0
        while replaced_pixels < num_pixels:
            perturbed_inputs = []

            batch = min(num_pixels - replaced_pixels, self.pixel_batch_size)

            # perturb # of pixel_batch_size pixels
            for pixel in range(batch):
                perturb_index = (
                    indices[0][replaced_pixels + pixel],
                    indices[1][replaced_pixels + pixel],
                )

                # perturb input using given pixels
                perturber.perturb(perturb_index[0], perturb_index[1])
            perturbed_inputs = perturber.get_current()
            replaced_pixels += batch

            # get score after perturb
            device = next(self.model.parameters()).device

            score_after = self.model(perturbed_inputs.to(device))
            score_after = score_after.view(-1)

            scores_after_perturb = np.concatenate(
                (scores_after_perturb, score_after.detach().cpu().numpy())
            )
        return scores_after_perturb
