# SmoothVideo

This repository is the official implementation of **Smooth Video Synthesis with Noise Constraints on Diffusion Models for One-shot Video Tuning**.




## Setup
This implementation is based on [Tune-A-Video](https://github.com/showlab/Tune-A-Video).

### Requirements

```shell
pip install -r requirements.txt
```

Installing [xformers](https://github.com/facebookresearch/xformers) is highly recommended for more efficiency and speed on GPUs. 
To enable xformers, set `enable_xformers_memory_efficient_attention=True` (default).

### Weights

**[Stable Diffusion]** [Stable Diffusion](https://arxiv.org/abs/2112.10752) is a latent text-to-image diffusion model capable of generating photo-realistic images given any text input. The pre-trained Stable Diffusion models can be downloaded from Hugging Face (e.g., [Stable Diffusion v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5))).


## Usage

### Training

To fine-tune the text-to-image diffusion models for text-to-video generation, run this command for the baseline model:
```bash
accelerate launch train_tuneavideo.py --config="configs/man-skiing.yaml"
```

Run this command for the baseline model with the proposed smooth loss:
```bash
accelerate launch train_tuneavideo.py --config="configs/man-skiing.yaml" --smooth_loss
```

Run this command for the baseline model with the proposed simple smooth loss:
```bash
accelerate launch train_tuneavideo.py --config="configs/man-skiing.yaml" --smooth_loss --simple_manner
```



Note: Tuning a 24-frame video usually takes `300~500` steps, about `10~15` minutes using one A100 GPU. 
Reduce `n_sample_frames` if your GPU memory is limited.

### Inference

Once the training is done, run inference:

```python
from tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from tuneavideo.models.unet import UNet3DConditionModel
from tuneavideo.util import save_videos_grid
import torch

pretrained_model_path = "./checkpoints/stable-diffusion-v1-5"
my_model_path = "./outputs/man-skiing"
unet = UNet3DConditionModel.from_pretrained(my_model_path, subfolder='unet', torch_dtype=torch.float16).to('cuda')
pipe = TuneAVideoPipeline.from_pretrained(pretrained_model_path, unet=unet, torch_dtype=torch.float16).to("cuda")
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_vae_slicing()

prompt = "spider man is skiing"
ddim_inv_latent = torch.load(f"{my_model_path}/inv_latents/ddim_latent-500.pt").to(torch.float16)
video = pipe(prompt, latents=ddim_inv_latent, video_length=24, height=512, width=512, num_inference_steps=50, guidance_scale=7.5).videos

save_videos_grid(video, f"./{prompt}.gif")
```

## Results
### Tune-A-Video
<table class="center">
<tr>
	<td > <b>Input video </b></td>
	<td colspan="4"><b> Tune-A-Video </b></td>
</tr>
<tr>
  <td><img src="./assets/tune-a-video/car-turn.gif"></td>
  <td><img src="./assets/tune-a-video/car-turn_a jeep car is moving on the beach.gif"></td>
    <td><img src="./assets/tune-a-video/car-turn_a jeep car is moving on the snow.gif"></td>
  <td><img src="./assets/tune-a-video/car-turn_a jeep car is moving on the road, cartoon style.gif"></td>
  <td><img src="./assets/tune-a-video/car-turn_a sports car is moving on the road.gif"></td>
</tr>
<tr>
	<td > <b> Input video </b> </td>
	<td colspan="4"><b> Tune-A-Video + smooth loss </b> </td>
</tr>
<tr>
<td><img src="./assets/tune-a-video/car-turn.gif"></td>
  <td><img src="./assets/tune-a-video/car-turn_smooth_a jeep car is moving on the beach.gif"></td>
    <td><img src="./assets/tune-a-video/car-turn_smooth_a jeep car is moving on the snow.gif"></td>
  <td><img src="./assets/tune-a-video/car-turn_smooth_a jeep car is moving on the road, cartoon style.gif"></td>
  <td><img src="./assets/tune-a-video/car-turn_smooth_a sports car is moving on the road.gif"></td>
</tr>
<tr>
	<td> a jeep car is moving on the road</td>
	<td> a jeep car is moving on the beach </td>
	<td> a jeep car is moving on the snow </td>
	<td> a jeep car is moving on the road, cartoon style </td>
	<td> a sports car is moving on the road </td>
</tr>
%
%
<tr>
	<td > <b>Input video </b></td>
	<td colspan="4"><b> Tune-A-Video </b></td>
</tr>
<tr>
<td><img src="./assets/tune-a-video/rabbit-watermelon.gif"></td>
  <td><img src="./assets/tune-a-video/rabbit-watermelon_a tiger is eating a watermelon.gif"></td>
    <td><img src="./assets/tune-a-video/rabbit-watermelon_a rabbit is eating an orange.gif"></td>
  <td><img src="./assets/tune-a-video/rabbit-watermelon_a rabbit is eating a pizza.gif"></td>
  <td><img src="./assets/tune-a-video/rabbit-watermelon_a puppy is eating an orange.gif"></td>
</tr>
<tr>
	<td > <b> Input video </b> </td>
	<td colspan="4"><b> Tune-A-Video + smooth loss </b> </td>
</tr>
<tr>
<td><img src="./assets/tune-a-video/rabbit-watermelon.gif"></td>
  <td><img src="./assets/tune-a-video/rabbit-watermelon_smooth_a tiger is eating a watermelon.gif"></td>
    <td><img src="./assets/tune-a-video/rabbit-watermelon_smooth_a rabbit is eating an orange.gif"></td>
  <td><img src="./assets/tune-a-video/rabbit-watermelon_smooth_a rabbit is eating a pizza.gif"></td>
  <td><img src="./assets/tune-a-video/rabbit-watermelon_smooth_a puppy is eating an orange.gif"></td>
</tr>
<tr>
	<td> a rabbit is eating a watermelon</td>
	<td> a tiger is eating a watermelon </td>
	<td> a rabbit is eating an orange </td>
	<td> a rabbit is eating a pizza </td>
	<td> a puppy is eating an orange</td>
</tr>
%
%
<tr>
	<td > <b>Input video </b></td>
	<td colspan="4"><b> Tune-A-Video </b></td>
</tr>
<tr>
<td><img src="./assets/tune-a-video/man-skiing.gif"></td>
  <td><img src="./assets/tune-a-video/man-skiing_mickey mouse is skiing on the snow.gif"></td>
    <td><img src="./assets/tune-a-video/man-skiing_spider man is skiing on the beach, cartoon style.gif"></td>
  <td><img src="./assets/tune-a-video/man-skiing_wonder woman, wearing a cowboy hat, is skiing.gif"></td>
  <td><img src="./assets/tune-a-video/man-skiing_a man, wearing pink clothes, is skiing at sunset.gif"></td>
</tr>
<tr>
	<td > <b> Input video </b> </td>
	<td colspan="4"><b> Tune-A-Video + smooth loss </b> </td>
</tr>
<tr>
<td><img src="./assets/tune-a-video/man-skiing.gif"></td>
  <td><img src="./assets/tune-a-video/man-skiing_smooth_mickey mouse is skiing on the snow.gif"></td>
    <td><img src="./assets/tune-a-video/man-skiing_smooth_spider man is skiing on the beach, cartoon style.gif"></td>
  <td><img src="./assets/tune-a-video/man-skiing_smooth_wonder woman, wearing a cowboy hat, is skiing.gif"></td>
  <td><img src="./assets/tune-a-video/man-skiing_smooth_a man, wearing pink clothes, is skiing at sunset.gif"></td>
</tr>
<tr>
	<td> a man is skiing</td>
	<td> mickey mouse is skiing on the snow </td>
	<td> spider man is skiing on the beach, cartoon style </td>
	<td> wonder woman, wearing a cowboy hat, is skiing </td>
	<td> a man, wearing pink clothes, is skiing at sunset </td>
</tr>





### Make-A-Protagonist

<table class="center">
<tr>
	<td > <b>Input video </b></td>
	<td ><b> Make-A-Protagonist </b></td>
	<td ><b> Make-A-Protagonist + smooth loss</b></td>
</tr>
<tr>
  <td><img src="./assets/make-a-protagonist/car-turn.gif"></td>
  <td><img src="./assets/make-a-protagonist/0-a jeep driving down a mountain road in the rain-org.gif"></td>
    <td><img src="./assets/make-a-protagonist/0-a jeep driving down a mountain road in the rain.gif"></td>
</tr>
<tr>
	<td> a jeep driving down a mountain road</td>
	<td> a jeep driving down a mountain road in the rain </td>
	<td> a jeep driving down a mountain road in the rain </td>
</tr>
<tr>
	<td><img src="./assets/make-a-protagonist/ikun.gif"></td>
  <td><img src="./assets/make-a-protagonist/0-A man is playing a basketball on the beach, anime style-org.gif"></td>
    <td><img src="./assets/make-a-protagonist/0-A man is playing a basketball on the beach, anime style.gif"></td>
</tr>
<tr>
	<td>A man is playing basketball</td>
	<td> A man is playing a basketball on the beach, anime style</td>
	<td> A man is playing a basketball on the beach, anime style </td>
</tr>
%
<tr>
<td><img src="./assets/make-a-protagonist/yanzi.gif"></td>
  <td><img src="./assets/make-a-protagonist/0-a panda walking down the snowy street-org.gif"></td>
    <td><img src="./assets/make-a-protagonist/0-a panda walking down the snowy street.gif"></td>
</tr>
<tr>
	<td>a man walking down the street at night</td>
	<td> a panda walking down the snowy street </td>
	<td> a panda walking down the snowy street </td>
</tr>
<tr>
<td><img src="./assets/make-a-protagonist/huaqiang.gif"></td>
  <td><img src="./assets/make-a-protagonist/0-elon musk walking down the street-org.gif"></td>
    <td><img src="./assets/make-a-protagonist/0-elon musk walking down the street.gif"></td>
</tr>
<tr>
	<td>a man walking down the street</td>
	<td> elon musk walking down the street </td>
	<td> elon musk walking down the street </td>
</tr>





### ControlVideo

<table class="center">
<tr>
	<td > <b>Input video </b></td>
	<td ><b> Condition </b></td>
	<td ><b> ControlVideo</b></td>
	<td ><b> ControlVideo + smooth loss</b></td>
</tr>
<tr>
  <td><img src="./assets/controlvideo/dance26-Michael Jackson is dancing-1500_merge_0.gif"></td>
<td><img src="./assets/controlvideo/dance26-Michael Jackson is dancing-1500_merge_1.gif"></td>
<td><img src="./assets/controlvideo/dance26-Michael Jackson is dancing-1500_merge_2.gif"></td>
<td><img src="./assets/controlvideo/dance26-Michael Jackson is dancing-1500_merge_3.gif"></td>
</tr>
<tr>
	<td>a person is dancing</td>
	<td> Pose </td>
	<td> Michael Jackson is dancing </td>
	<td> Michael Jackson is dancing </td>
</tr>
%
<tr>
  <td><img src="./assets/controlvideo/dance5-a person is dancing, Makoto Shinkai style-1000_merge_0.gif"></td>
<td><img src="./assets/controlvideo/dance5-a person is dancing, Makoto Shinkai style-1000_merge_1.gif"></td>
<td><img src="./assets/controlvideo/dance5-a person is dancing, Makoto Shinkai style-1000_merge_2.gif"></td>
<td><img src="./assets/controlvideo/dance5-a person is dancing, Makoto Shinkai style-1000_merge_3.gif"></td>
</tr>
<tr>
	<td>a person is dancing</td>
	<td> Pose </td>
	<td> a person is dancing, Makoto Shinkai style </td>
	<td> a person is dancing, Makoto Shinkai style</td>
</tr>
%
<tr>
  <td><img src="./assets/controlvideo/building1-a wooden building, at night-80_merge_0.gif"></td>
<td><img src="./assets/controlvideo/building1-a wooden building, at night-80_merge_1.gif"></td>
<td><img src="./assets/controlvideo/building1-a wooden building, at night-80_merge_2.gif"></td>
<td><img src="./assets/controlvideo/building1-a wooden building, at night-80_merge_3.gif"></td>
</tr>
<tr>
	<td>a building</td>
	<td> Canny edge </td>
	<td> a wooden building, at night </td>
	<td> a wooden building, at night</td>
</tr>
%
<tr>
  <td><img src="./assets/controlvideo/girl8-a girl, Krenz Cushart style-300_merge_0.gif"></td>
<td><img src="./assets/controlvideo/girl8-a girl, Krenz Cushart style-300_merge_1.gif"></td>
<td><img src="./assets/controlvideo/girl8-a girl, Krenz Cushart style-300_merge_2.gif"></td>
<td><img src="./assets/controlvideo/girl8-a girl, Krenz Cushart style-300_merge_3.gif"></td>
</tr>
<tr>
	<td>a girl</td>
	<td> Hed edge </td>
	<td> a girl, Krenz Cushart style</td>
	<td> a girl, Krenz Cushart style</td>
</tr>
%
<tr>
  <td><img src="./assets/controlvideo/girlface9_6-a girl with rich makeup-300_merge_0.gif"></td>
<td><img src="./assets/controlvideo/girlface9_6-a girl with rich makeup-300_merge_1.gif"></td>
<td><img src="./assets/controlvideo/girlface9_6-a girl with rich makeup-300_merge_2.gif"></td>
<td><img src="./assets/controlvideo/girlface9_6-a girl with rich makeup-300_merge_3.gif"></td>
</tr>
<tr>
	<td>a girl</td>
	<td> Hed edge </td>
	<td> a girl with rich makeup</td>
	<td> a girl with rich makeup</td>
</tr>
%
<tr>
  <td><img src="./assets/controlvideo/ink1-gentle green ink diffuses in water, beautiful light-200_merge_0.gif"></td>
<td><img src="./assets/controlvideo/ink1-gentle green ink diffuses in water, beautiful light-200_merge_1.gif"></td>
<td><img src="./assets/controlvideo/ink1-gentle green ink diffuses in water, beautiful light-200_merge_2.gif"></td>
<td><img src="./assets/controlvideo/ink1-gentle green ink diffuses in water, beautiful light-200_merge_3.gif"></td>
</tr>
<tr>
	<td>ink diffuses in water</td>
	<td> Depth </td>
	<td> gentle green ink diffuses in water, beautiful light</td>
	<td> gentle green ink diffuses in water, beautiful light</td>
</tr>




### Video2Video-zero

<table class="center">
<tr>
	<td > <b>Input video </b></td>
	<td ><b> InstructVideo2Video-zero </b></td>
	<td ><b> InstructVideo2Video-zero+noise constraint </b></td>
	<td ><b> Video InstructPix2Pix </b></td>
	<td ><b> Video InstructPix2Pix+noise constraint </b></td>
</tr>
<tr>
	<td><img src="./assets/video2video-zero/mini-cooper.gif"></td>
  	<td><img src="./assets/video2video-zero/mini-cooper_make it animation_InstructVideo2Video-zero.gif"></td>
    	<td><img src="./assets/video2video-zero/mini-cooper_make it animation_InstructVideo2Video-zero_noise_cons.gif"></td>
  	<td><img src="./assets/video2video-zero/mini-cooper_make it animation_Video-InstructPix2Pix.gif"></td>
    	<td><img src="./assets/video2video-zero/mini-cooper_make it animation_Video-InstructPix2Pix_noise_cons.gif"></td>
</tr>
<tr>
	<td colspan="3"> <instruct make it animation></td>
</tr>
</table>


</table>




## Shoutouts

- This code builds on [Tune-A-Video](https://github.com/showlab/Tune-A-Video) and [diffusers](https://github.com/huggingface/diffusers). Thanks for open-sourcing!
