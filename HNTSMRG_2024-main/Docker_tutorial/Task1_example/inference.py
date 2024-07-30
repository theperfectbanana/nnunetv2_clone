"""
The following is a simple example algorithm for Task 1 (pre-RT segmentation) of the HNTS-MRG 2024 challenge.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To export the container and prep it for upload to Grand-Challenge.org you can call:

  docker save example-algorithm-task-1-pre-rt-segmentation | gzip -c > example-algorithm-task-1-pre-rt-segmentation.tar.gz

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""

from pathlib import Path
from glob import glob
import SimpleITK as sitk
import numpy as np
from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

INPUT_PATH = Path("/input") # these are the paths that Docker will use
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")

def run():
    """
    Main function to read input, process the data, and write output.
    """
    
    ### 1. Read the input from the correct path. 
    # Currently reads input as a SimpleITK image. 
    location = INPUT_PATH / "images/pre-rt-t2w-head-neck", # Make sure to read from this path; this is exactly how the Grand Challenge will give the input to your container.
    
    
    ### 2. Process the inputs any way you'd like. 
    # This is where you should place your relevant inference code.
    _show_torch_cuda_info()

    with open(RESOURCE_PATH / "some_resource.txt", "r") as f:
        print(f.read())
    
    ### 3. Save your output to the correct path. 
    # Currently takes generated mask (in SimpleITK image format) and writes to MHA file. 
    location = OUTPUT_PATH / "images/mri-head-neck-segmentation", # Make sure to write to this path; this is exactly how the Grand Challenge will expect the output from your container.
    
    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )

 # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, 'Dataset003_Liver/nnUNetTrainer__nnUNetPlans__3d_lowres'),
        use_folds=(0,),
        checkpoint_name='checkpoint_final.pth',
    )

    # variant 1: give input and output folders
    predictor.predict_from_files(join(nnUNet_raw, 'INPUT PATH'),
                                 join(nnUNet_raw, 'OUTPUT PATH'),
                                 save_probabilities=False, overwrite=False,
                                 num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                 folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
    
    return 0



def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())