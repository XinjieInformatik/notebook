# Diffusers

## Document
https://huggingface.co/docs/diffusers/v0.25.1/en/index

### Effective-and-efficient-diffusion
https://huggingface.co/docs/diffusers/v0.25.1/en/stable_diffusion#effective-and-efficient-diffusion

better infer speed:
1. fp32 -> fp16
2. switch performant scheduler
3. enable_attention_slicing

better quality:
1. better checkpoints (https://huggingface.co/models?library=diffusers&sort=downloads / https://huggingface.co/spaces/huggingface-projects/diffusers-gallery)
2. better pipeline componets. e.g. latest autoencoder
3. better prompt engineering
    - How is the image or similar images of the one I want to generate stored on the internet?
    - What additional detail can I give that steers the model towards the style I want?


```python
>>> prompt = "portrait photo of a old warrior chief"
>>> prompt += ", tribal panther make up, blue on red, side profile, looking away, serious eyes"
>>> prompt += " 50mm portrait photography, hard rim lighting photography--beta --ar 2:3  --beta --upbeta"
prompt_zh = "老战士酋长的肖像照片，部落豹妆，红底蓝，侧面轮廓，看向别处，严肃的眼睛 50 毫米肖像摄影，硬框照明摄影--beta --ar 2:3 --beta --upbeta"
```




