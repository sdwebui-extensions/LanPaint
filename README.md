# LanPaint (Thinking mode Inpaint)

Unlock precise inpainting without additional training. LanPaint lets the model "think" through multiple iterations before denoising, aiming for seamless and accurate results.


We encourage you to try it out and share your feedback through issues or discussions, as your input will help us enhance the algorithm's performance and stability.

## Features

- 🎨 **Zero-Training Inpainting** - Works immediately with ANY SD model, even custom models you've trained yourself
- 🛠️ **Simple Integration** - Same workflow as standard ComfyUI KSampler
- 🎯 **True Blank-Slate Generation** - No need to set default denoise at 0.7 (preserving 30% original pixels in masks) used in conventional methods: 100% **new content creation**, No "painting over" existing content.
- 🌈 **Not only inpaint**: You can even use it as a simple way to generate consistent characters.

## How It Works? 
- 🤝 **Bidirectional Context Control** 
    - Creates **two-way alignment** between masked/unmaksed areas  
    - Unlike single-direction approaches, our system continuously checks:  
      *"Does the new content make sense with existing elements?"*  
      *"Do existing elements support the new creation?"*  


## Key Differences at a Glance  

|                      | Common Tools | LanPaint |  
|----------------------|--------------|----------|  
| Works with any SD model | ❌           | ✅       |  
| Context awareness    | ❌ One-way   | ✅ Two-way |  

## Updates

LanPaint has received a major update! All examples now use the LanPaint K Sampler, offering a simplified interface with enhanced performance and stability.

## Example Results
All examples use a random seed 0 to generate batch of 4 images for fair comparison. (Warning: Generating 4 images may exceed your GPU memory; adjust batch size as needed.)

### Example 1: Basket to Basket Ball (LanPaint K Sampler, 2 steps of thinking).
![Inpainting Result 1](https://github.com/scraed/LanPaint/blob/master/examples/InpaintChara_04.jpg)  
[View Workflow & Masks](https://github.com/scraed/LanPaint/tree/master/examples/Example_1) 
[Model Used in This Example](https://civitai.com/models/1188071?modelVersionId=1408658) 
### Example 2: White Shirt to Blue Shirt (LanPaint K Sampler, 5 steps of thinking)
![Inpainting Result 2](https://github.com/scraed/LanPaint/blob/master/examples/InpaintChara_05.jpg)  
[View Workflow & Masks](https://github.com/scraed/LanPaint/tree/master/examples/Example_2)
[Model Used in This Example](https://civitai.com/models/1188071?modelVersionId=1408658)
### Example 3: Smile to Sad (LanPaint K Sampler, 5 steps of thinking)
![Inpainting Result 3](https://github.com/scraed/LanPaint/blob/master/examples/InpaintChara_06.jpg)  
[View Workflow & Masks](https://github.com/scraed/LanPaint/tree/master/examples/Example_3)
[Model Used in This Example](https://civitai.com/models/133005/juggernaut-xl)
### Example 4: Damage Restoration (LanPaint K Sampler, 5 steps of thinking)
![Inpainting Result 4](https://github.com/scraed/LanPaint/blob/master/examples/InpaintChara_07.jpg)  
[View Workflow & Masks](https://github.com/scraed/LanPaint/tree/master/examples/Example_4)
[Model Used in This Example](https://civitai.com/models/133005/juggernaut-xl)
### Example 5: Huge Damage Restoration (LanPaint K Sampler, 20 steps of thinking)
![Inpainting Result 5](https://github.com/scraed/LanPaint/blob/master/examples/InpaintChara_08.jpg)  
[View Workflow & Masks](https://github.com/scraed/LanPaint/tree/master/examples/Example_5)
[Model Used in This Example](https://civitai.com/models/133005/juggernaut-xl)
### Example 6: Character Consistency (Side View Generation) (LanPaint K Sampler, 5 steps of thinking)
![Inpainting Result 6](https://github.com/scraed/LanPaint/blob/master/examples/InpaintChara_09.jpg)  
[View Workflow & Masks](https://github.com/scraed/LanPaint/tree/master/examples/Example_6)
[Model Used in This Example](https://civitai.com/models/1188071?modelVersionId=1408658) 

(Tricks 1: You can emphasize the character by copy it's image multiple times with Photoshop. Here I have made one extra copy.)

(Tricks 2: Use prompts like multiple views, multiple angles, clone, turnaround.)

**How to Use These Examples:**  
1. Navigate to the **example** folder (i.e example_1) by clicking **View Workflow & Masks**, download all pictures.  
2. Drag **InPainted_Drag_Me_to_ComfyUI.png** into ComfyUI to load the workflow.  
3. Download the required model from Civitai by clicking **Model Used in This Example**.  
4. Load the model into the **"Load Checkpoint"** node.  
5. Upload **Original_No_Mask.png** to the **"Load image"** node in the **"Original Image"** group (far left).  
6. Upload **Masked_Load_Me_in_Loader.png** to the **"Load image"** node in the **"Mask image for inpainting"** group (second from left).  
7. Queue the task, you will get inpainted results from three methods:  
   - **[VAE Encode for Inpainting](https://comfyanonymous.github.io/ComfyUI_examples/inpaint/)** (middle),  
   - **[Set Latent Noise Mask](https://comfyui-wiki.com/en/tutorial/basic/how-to-inpaint-an-image-in-comfyui)** (second from right),  
   - **LanPaint** (far right).  

Compare and explore the results from each method!

![WorkFlow](https://github.com/scraed/LanPaint/blob/master/Example.JPG)  

## Quickstart

1. **Install ComfyUI**: Follow the official [ComfyUI installation guide](https://docs.comfy.org/get_started) to set up ComfyUI on your system.  
2. **Install ComfyUI-Manager**: Add the [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) for easy extension management.  
3. **Install LanPaint Nodes**:  
   - **Via ComfyUI-Manager**: Search for "[LanPaint](https://registry.comfy.org/publishers/scraed/nodes/LanPaint)" in the manager and install it directly.  
   - **Manually**: Click "Install via Git URL" in ComfyUI-Manager and input the GitHub repository link:  
     ```
     https://github.com/scraed/LanPaint.git
     ```  
     Alternatively, clone this repository into the `ComfyUI/custom_nodes` folder.  
4. **Restart ComfyUI**: Restart ComfyUI to load the LanPaint nodes.  

Once installed, you'll find the LanPaint nodes under the "sampling" category in ComfyUI. Use them just like the default KSampler for high-quality inpainting!

## Usage

**Workflow Setup**  
Same as default ComfyUI KSampler - simply replace with LanPaint KSampler nodes. The inpainting workflow is the same as the [SetLatentNoiseMask](https://comfyui-wiki.com/zh/comfyui-nodes/latent/inpaint/set-latent-noise-mask) inpainting workflow.

**Note**
- LanPaint requires binary masks (values of 0 or 1) without opacity or smoothing. To ensure compatibility, set the mask's **opacity and hardness to maximum** in your mask editor. During inpainting, any mask with smoothing or gradients will automatically be converted to a binary mask.
- LanPaint relies heavily on your text prompts to guide inpainting - explicitly describe the content you want generated in the masked area. If results show artifacts or mismatched elements, counteract them with targeted negative prompts.

## Basic Sampler
![Samplers](https://github.com/scraed/LanPaint/blob/master/Nodes.JPG)  
### LanPaint KSampler
Simplified interface with recommended defaults:

- Steps: 50+ recommended
- LanPaint NumSteps: The turns of thinking before denoising. Recommend 5 for most of tasks.
- LanPaint EndSigma: The noise level below which thinking is disabled. Recommend 0.6 for realistic style (tested on Juggernaut-xl), 3.0 for anime style (tested on Animagine XL 4.0)


### LanPaint KSampler (Advanced)
Full parameter control:
**Key Parameters**

| Parameter | Range | Description |
|-----------|-------|-------------|
| `Steps` | 0-100 | Total steps of diffusion sampling. Higher means better inpainting. Recommend 50. |
| `LanPaint_NumSteps` | 0-20 | Reasoning iterations per denoising step ("thinking depth"). Easy task: 1-2. Hard task: 5-10 |
| `LanPaint_Lambda` | 0.1-50 | Content alignment strength (higher = stricter). Recommend 8.0 |
| `LanPaint_StepSize` | 0.1-1.0 | The StepSize of each thinking step. Recommend 0.5. |
| `LanPaint_cfg_BIG` | -20-20 | CFG scale used when aligning masked and unmasked region (positive value tends to ignores promts, negative value enhances prompts.). Recommend 8 for seamless inpaint (i.e limbs, faces) when prompt is not important. -0.5 when prompt is important, like character consistency (i.e multiple view) |

For detailed descriptions of each parameter, simply hover your mouse over the corresponding input field to view tooltips with additional information.



## LanPaint KSampler (Advanced) Tuning Guide
For challenging inpainting tasks:  

1️⃣ **Primary Adjustments**:
- Increase **steps**, **LanPaint_StepSize**，**LanPaint_NumSteps** (thinking iterations), and **LanPaint_cfg_BIG** (cfg scale used for unmasked region ).
  
2️⃣ **Secondary Tweaks**:  
- Boost **LanPaint_Lambda** (bidirectional guidance scale) or **LanPaint_StepSize** (denoising aggressiveness).
    
3️⃣ **Balance Speed vs Stability**:  
- Reduce **LanPaint_Friction** to prioritize faster results with fewer "thinking" steps (*may risk instability*).  
- Increase **LanPaint_Tamed** (noise normalization onto a sphere) or **LanPaint_Alpha** (constraint the friction of underdamped Langevin dynamics) to suppress artifacts.

⚠️ **Notes**:  
- Optimal parameters vary depending on the **model** and the **size of the inpainting area**.  
- For effective tuning, **fix the seed** and adjust parameters incrementally while observing the results. This helps isolate the impact of each setting.  

## Contribute

- 2025/03/06: Bug Fix for str not callable error and unpack error. Big thanks to [jamesWalker55](https://github.com/jamesWalker55) and [EricBCoding](https://github.com/EricBCoding).

Help us improve LanPaint! 🚀 **Report bugs**, share **example cases**, or contribute your **personal parameter settings** to benefit the community. 





