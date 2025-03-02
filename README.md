# LanPaint

Powerful Training Free Inpainting Tool Works for Every SD Model.

## Features

- 🎨 **Zero-Training Inpainting** - Works immediately with ANY SD model, even custom models you've trained yourself
- 🛠️ **Simple Integration** - Same workflow as standard ComfyUI KSampler
- 🚀 **Quality Enhancements** - High quality and seamless inpainting

- 
![Node Comparison](https://github.com/scraed/LanPaint/blob/master/examples/InpaintChara_04.jpg)
![Node Comparison](https://github.com/scraed/LanPaint/blob/master/examples/InpaintChara_05.jpg)
![Node Comparison](https://github.com/scraed/LanPaint/blob/master/examples/InpaintChara_06.jpg)

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


## Develop

To install the dev dependencies and pre-commit (will run the ruff hook), do:

```bash
cd LanPaint
pip install -e .[dev]
pre-commit install
```

The `-e` flag above will result in a "live" install, in the sense that any changes you make to your node extension will automatically be picked up the next time you run ComfyUI.

## Publish to Github

Install Github Desktop or follow these [instructions](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) for ssh.

1. Create a Github repository that matches the directory name. 
2. Push the files to Git
```
git add .
git commit -m "project scaffolding"
git push
``` 

## Writing custom nodes

An example custom node is located in [node.py](src/LanPaint/nodes.py). To learn more, read the [docs](https://docs.comfy.org/essentials/custom_node_overview).


## Tests

This repo contains unit tests written in Pytest in the `tests/` directory. It is recommended to unit test your custom node.

- [build-pipeline.yml](.github/workflows/build-pipeline.yml) will run pytest and linter on any open PRs
- [validate.yml](.github/workflows/validate.yml) will run [node-diff](https://github.com/Comfy-Org/node-diff) to check for breaking changes

## Publishing to Registry

If you wish to share this custom node with others in the community, you can publish it to the registry. We've already auto-populated some fields in `pyproject.toml` under `tool.comfy`, but please double-check that they are correct.

You need to make an account on https://registry.comfy.org and create an API key token.

- [ ] Go to the [registry](https://registry.comfy.org). Login and create a publisher id (everything after the `@` sign on your registry profile). 
- [ ] Add the publisher id into the pyproject.toml file.
- [ ] Create an api key on the Registry for publishing from Github. [Instructions](https://docs.comfy.org/registry/publishing#create-an-api-key-for-publishing).
- [ ] Add it to your Github Repository Secrets as `REGISTRY_ACCESS_TOKEN`.

A Github action will run on every git push. You can also run the Github action manually. Full instructions [here](https://docs.comfy.org/registry/publishing). Join our [discord](https://discord.com/invite/comfyorg) if you have any questions!

