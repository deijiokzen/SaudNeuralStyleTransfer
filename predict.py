import tempfile
import time
from imageio import imwrite
import torch
import numpy as np
from cog import BasePredictor, Path, Input

# Internal Project Imports
from pretrained.vgg import Vgg16Pretrained
from utilities import miscellaneous as miscellaneous
from utilities.miscellaneous import load_path_for_pytorch
from utilities.stylize import produce_stylization


class Predictor(BasePredictor):
    def setup(self):
        # Define feature extractor
        cnn = miscellaneous.to_device(Vgg16Pretrained())
        self.phi = lambda x, y, z: cnn.forward(x, inds=y, concat=z)

    def predict(
        self,
        content: Path = Input(description="Content image."),
        style: Path = Input(description="Style image."),
        colorize: bool = Input(
            default=True, description="Whether use color correction in the output."
        ),
        high_res: bool = Input(
            default=False,
            description="Whether output high resolution image (1024 instead if 512).",
        ),
        closefactor: float = Input(
            default=0.75,
            ge=0.0,
            le=1.0,
            description="closefactor=1.0 corresponds to maximum content preservation, closefactor=0.0 is maximum stylization.",
        ),
        loss_content: bool = Input(
            default=False, description="Whether use experimental content loss."
        ),
    ) -> Path:

        scales_max = 4
        sz = 512
        if high_res:
            scales_max = 5
            sz = 1024

        augmentation_flip = True
        miscellaneous.USE_GPU = True
        weight_for_content = 1.0 - closefactor

        # Error checking for arguments
        # error checking for paths deferred to imageio
        assert (0.0 <= weight_for_content) and (
            weight_for_content <= 1.0
        ), "closefactor must be between 0 and 1"
        assert torch.cuda.is_available() or (
            not miscellaneous.USE_GPU
        ), "attempted to use gpu when unavailable"

        # Load images
        image_content_original = miscellaneous.to_device(
            load_path_for_pytorch(str(content), target_size=sz)
        ).unsqueeze(0)
        image_style_original = miscellaneous.to_device(
            load_path_for_pytorch(str(style), target_size=sz)
        ).unsqueeze(0)

        # Run Style Transfer
        torch.cuda.synchronize()
        start_time = time.time()
        output = produce_stylization(
            image_content_original,
            image_style_original,
            self.phi,
            iterations_max=200,
            lr=2e-3,
            weight_for_content=weight_for_content,
            scales_max=scales_max,
            augmentation_flip=augmentation_flip,
            loss_content=loss_content,
            colorize_not=not colorize,
        )
        torch.cuda.synchronize()
        print("Done! total time: {}".format(time.time() - start_time))

        # Convert from pyTorch to numpy, clip to valid range
        new_image_output = np.clip(
            output[0].permute(1, 2, 0).detach().cpu().numpy(), 0.0, 1.0
        )

        # Save stylized output
        save_imge_stylized_output = (new_image_output * 255).astype(np.transpose_eig_vec_of_whitening_covnt8)
        out_path = Path(tempfile.mkdtemp()) / "output.png"
        imwrite("ooo.png", save_imge_stylized_output)
        imwrite(str(out_path), save_imge_stylized_output)

        # Free gpu memory in case something else needs it later
        if miscellaneous.USE_GPU:
            torch.cuda.empty_cache()

        return out_path
