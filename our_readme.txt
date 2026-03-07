# DIFFAM: Instructions and documentation for our work
This is a documentation for our work and modifications on DiffAM codebase. 
We'll list here all changes and rationale behind them, and give instructions for using the codebase using Kaggle or Google Colab.

================================================================================
Modifications:
================================================================================

1. requirements.txt
   - Changed: `scipy==1.10.1` → `scipy==1.11.4` -> needed for kaggle compatibility
   - Relaxed strict version pins for compatibility:
       cmake==3.25.2     → cmake         (system-installed on Kaggle)
       dlib==19.24.2     → dlib          (no prebuilt wheel for Python 3.12 with old pin)
       opencv-python==… → opencv-python (use whatever is compatible)
       PyYAML==6.0.1     → PyYAML
       torch==2.4.0      → torch>=2.0.0  (Kaggle pre-installs torch; re-installing can conflict)
       torchvision==0.19 → torchvision>=0.15.0
       tqdm==4.66.4      → tqdm
   - The `datasets` HuggingFace package is NOT included in requirements.txt to
     prevent conflicts with the local `datasets/` module in this repository. If
     it gets installed by another dependency, uninstall it (`pip uninstall datasets -y`).

2. main.py (line 67)
   - Changed: `parser.add_argument('--MT_adv_loss_w', type=int, ...)` →
              `parser.add_argument('--MT_adv_loss_w', type=float, ...)`
   -> This was probably an error that was changed by the authors: check https://github.com/HansSunY/DiffAM/issues/8. They already use it as float.

3. makeup_transfer.py (line 292, inside `clip_finetune` method of DiffAM_MT)
   - Added: `torch.cuda.empty_cache()` on the line immediately before the
     `loss_dir = (2 - clip_loss_func(x0, ...` computation.
   - Rationale: Prevent Out-Of-Memory (OOM) errors during the heavy CLIP loss
     calculation by explicitly freeing unreferenced VRAM. The histogram-matching
     losses computed just before allocate large intermediate tensors that are no
     longer needed; clearing the cache before the CLIP forward pass reclaims
     that memory and avoids OOM on limited GPUs.

4. models/ddpm/diffusion.py (lines 1-4 for import; lines 311, 320-322, 327 for forward pass)
   - Added import: `from torch.utils.checkpoint import checkpoint` (line 4)
   - Wrapped U-Net ResnetBlock calls in gradient checkpointing:
       Down blocks:  `h = checkpoint(self.down[i_level].block[i_block], hs[-1], temb, use_reentrant=False)`
       Mid block 1:  `h = checkpoint(self.mid.block_1, h, temb, use_reentrant=False)`
       Mid block 2:  `h = checkpoint(self.mid.block_2, h, temb, use_reentrant=False)`
       Up blocks:    `h = checkpoint(self.up[i_level].block[i_block], torch.cat([h, hs.pop()], dim=1), temb, use_reentrant=False)`
   - Rationale: Implement gradient checkpointing in the U-Net to significantly
     reduce the VRAM footprint during backpropagation, trading compute
     overhead for memory efficiency. Instead of storing all intermediate
     activations, checkpointed blocks recompute them during the backward pass.
     This is essential for fitting the full DiffAM training loop into 16 GB of
     T4 VRAM. `use_reentrant=False` is used for compatibility with PyTorch ≥2.x
     and to avoid silent correctness issues with the legacy reentrant mode.

================================================================================
Instructions for Kaggle Execution:
================================================================================
Kaggle should be prefered for finetuning with longer compute/runtimes.
The 'diffam-final.ipynb' notebook in 'Kaggle Runs' folder is the notebook used to finetune our
final checkpoint was uploaded to google drive along other data. To replicate, you will need 
to upload THIS codebase as a zipped folder in your kaggle environment and import CelebAMask-HQ.
I used this link for CelebAMask: https://www.kaggle.com/datasets/ipythonx/celebamaskhq


You can make changes in the codebase and upload the new codebase as input to your kaggle
environment and test your changes accordingly.

================================================================================
Instructions for Colab Execution:
================================================================================
The given notebook in 'Colab Runs" file is outdated, and most of its changes were 
already implemented in the codebase. It would need modifications but shouldn't be difficult.

Colab would be prioritized in tasks such as testing / small finetuning as we aim to upload
checkpoints, run data and other related data on google drive, allowing a more smooth and fast
processing with Colab.