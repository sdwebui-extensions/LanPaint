# LanPaint

Powerful Training Free Inpainting Tool Works for Every SD Model. Official Implementation of "Lanpaint: Training-Free Diffusion Inpainting with Exact and Fast Conditional Inference".

## Features

- 🎨 **Zero-Training Inpainting** - Works immediately with ANY SD model, even custom models you've trained yourself
- 🛠️ **Simple Integration** - Same workflow as standard ComfyUI KSampler
- 🚀 **Quality Enhancements** - High quality and seamless inpainting

## Example Results

### Example 1 (LanPaint K Sampler)
![Inpainting Result 1](https://github.com/scraed/LanPaint/blob/master/examples/InpaintChara_04.jpg)  
[View Workflow & Masks](https://github.com/scraed/LanPaint/tree/master/examples/Example_1)

### Example 2: (LanPaint K Sampler (Advanced))
![Inpainting Result 2](https://github.com/scraed/LanPaint/blob/master/examples/InpaintChara_05.jpg)  
[View Workflow & Masks](https://github.com/scraed/LanPaint/tree/master/examples/Example_2)

### Example 3: (LanPaint K Sampler (Advanced))
![Inpainting Result 3](https://github.com/scraed/LanPaint/blob/master/examples/InpaintChara_06.jpg)  
[View Workflow & Masks](https://github.com/scraed/LanPaint/tree/master/examples/Example_3)

Each example includes:
- Original masked image
- Full ComfyUI workflow


## Quickstart

1. Install [ComfyUI](https://docs.comfy.org/get_started).
1. Install [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)
1. Look up this extension in ComfyUI-Manager. If you are installing manually, clone this repository under `ComfyUI/custom_nodes`.
1. Restart ComfyUI.


## Installation

1. Place `LanPaint_Nodes.py` in your `ComfyUI/custom_nodes` folder
2. Restart ComfyUI
3. Use like regular KSampler with inpainting workflows

## Usage

**Workflow Setup**  
Same as default ComfyUI KSampler - simply replace with LanPaint KSampler nodes. The inpainting workflow is the same as the [SetLatentNoiseMask](https://comfyui-wiki.com/zh/comfyui-nodes/latent/inpaint/set-latent-noise-mask) inpainting workflow.

**Note**
LanPaint only support binary mask (0,1) with no smoothing. Any mask with smooting will be converted to binary mask during inpainting.



## Advanced Options (Optional)

Fine-tune results with these key parameters:

| Parameter | Typical Range | Effect |
|-----------|---------------|--------|
| `NumSteps` | 1-10 | Thinking iterations per step |
| `Lambda` | 4-8 | Content preservation strength |
| `StepSize` | 0.05-0.2 | Detail refinement intensity |

## Citation

If you use LanPaint in your research or projects, please cite our work:

```bibtex
@misc{zheng2025lanpainttrainingfreediffusioninpainting,
      title={Lanpaint: Training-Free Diffusion Inpainting with Exact and Fast Conditional Inference}, 
      author={Candi Zheng and Yuan Lan and Yang Wang},
      year={2025},
      eprint={2502.03491},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2502.03491}, 
}
```



