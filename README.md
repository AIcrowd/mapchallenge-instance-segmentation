
## Installation from Source

```bash
# Clone the repository
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection

# Create and activate a conda environment
conda create -n openmmlab python=3.8 -y
conda activate openmmlab

# Install PyTorch (adjust according to your CUDA version)
pip install torch torchvision torchaudio

# Install MMDetection from source
pip install -v -e .

# Install additional dependencies
pip install mmengine mmcv
```

## Running MAP Challenge Instance Segmentation Experiments

### Project Configuration
You can find and edit experiment configurations in the `projects/mapchallenge` folder. Typical configurations will include:
- Model architectures
- Dataset settings
- Training parameters

### Experiment Script
```bash
# Run the experiments script
bash train.sh
```
## MAP Challenge Instance Segmentation Results

| Model            | segm mAP | segm mAP_50 | segm mAP_75 | segm mAP_s | segm mAP_m | segm mAP_l | segm mAR | segm mAR_50 | segm mAR_75 |
|------------------|----------|-------------|-------------|------------|------------|------------|---------|--------|------------|
| SwinS-Mask2former| 0.3110   | 0.7160      | 0.2180      | 0.1690     | 0.4420     | 0.0990     | 0.42    | 0.833  | 0.362      |
| SwinL-Mask2former| 0.3080   | 0.7260      | 0.2360      | 0.1710     | 0.4400     | 0.1310     | 0.453   | 0.855  | 0.464      |
| Rtmdet-X         | 0.3910   | 0.7760      | 0.3440      | 0.2090     | 0.5440     | 0.2930     | 0.509   | 0.891  | 0.5        |
| RTMdet-M         | 0.3790   | 0.7540      | 0.3170      | 0.2150     | 0.5210     | 0.2820     | 0.485   |  0.87  | 0.457      |
| QueryInst-r50    | 0.2770   | 0.6520      | 0.1830      | 0.1380     | 0.3890     | 0.1530     | 0.432   | 0.826  | 0.384      |
| QueryInst-r101   | 0.2780   | 0.6340      | 0.2000      | 0.1760     | 0.3910     | 0.1010     | 0.458   | 0.855  | 0.464      |
| [**MaskDINO**](https://github.com/AIcrowd/MaskDINO-mapchallenge/blob/main/logs/experiment/maskdino-v2-full-run-1xH100-maskdino_R50_bs16_50ep_4s_dowsample1_2048.txt) | **0.584**| **0.9022**| **0.6150**| **0.3670**| **0.6911**| **0.9287**| **0.6802**| **0.9569**| **0.75**|


<br>

## MAP Challenge Results for 0.50 IoU

| Model            | segm mAP_50 |
|------------------|-------------|
| SwinS-Mask2former| 0.7160      |
| SwinL-Mask2former| 0.7260      |
| Rtmdet-X         | 0.7760      |
| RTMdet-M         | 0.7540      |
| QueryInst-r50    | 0.6520      |
| QueryInst-r101   | 0.6340      |
| **MaskDINO**     | **0.9022**   |

<br>
<!-- For the MAP Challenge instance segmentation task, **MaskDINO** got the most robust and high-performing model, especially for small and large objects.
 -->
## Related Repositories

For MaskDINO experiments, you might want to check out:
- [MaskDINO training Repository](https://github.com/saidineshpola/MaskDINO-mapchallenge/blob/main/README.md)

