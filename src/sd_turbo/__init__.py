import torch
from diffusers import AutoPipelineForText2Image

print("Cargando Modelo...")

modelo = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo",
     torch_dtype=torch.float32
     #FloatXX 32 o 16
     # Me indica la cantidad de decimales de precision
     # con el modelo que creara la imagen
     # Si mi pc me generera una imagen negra, cambio a 16
 )

modelo = modelo.to("cpu")
#cuda - nvidia
# mps - Mac con procesadores MX
# xpu - Grafica Intel IRIS

prompt = input("Escribe el promt de la imagen: ")
negative_prompt = """worst quality, low quality:1.4, blur, blurry, bokeh:1.2, compression artifacts, bad anatomy,
 deformed limbs, poorly drawn hands, extra fingers,
 mutation, disfigured, ugly, text, watermark, signature, logo, cropped."""

print("Generando la imagen ...")

imagen = modelo(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inferencias_steps=10, # Nuemro de veces que creara y corregira la imagen
    guidance_scale=2.0, # Que tan fiel sera al prompt
    height=1024,
    widht=1024,
).images[0]

imagen.save("imagen.png")
print("Imagen guardada !")
