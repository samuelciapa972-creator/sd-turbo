# Texto a imagen con SD-Turbo

Proyecto de práctica de **IA Generativa** que genera imágenes a partir de un texto (*text-to-image*) usando el modelo [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo) de Stability AI con la librería [Diffusers](https://github.com/huggingface/diffusers) de Hugging Face. El script pide el prompt por consola y guarda el resultado como `imagen.png`.

## Requisitos

- Python 3.12
- [uv](https://docs.astral.sh/uv/) para gestionar el entorno y las dependencias
- Conexión a internet la primera vez (para descargar el modelo desde Hugging Face)
- El modelo se ejecuta en **CPU**, por lo que no hace falta GPU (aunque la generación es más lenta)

## Instalación

```bash
git clone https://github.com/samuelciapa972-creator/sd-turbo.git
cd sd-turbo
uv sync
```

`uv sync` crea el entorno virtual `.venv` e instala las dependencias definidas en `pyproject.toml`:

| Paquete        | Versión   | Uso                                                        |
|----------------|-----------|------------------------------------------------------------|
| `diffusers`    | >= 0.41.0 | Pipeline de texto a imagen (`AutoPipelineForText2Image`)   |
| `torch`        | >= 2.14.1 | Motor de cálculo (tensores) sobre el que corre el modelo   |
| `transformers` | >= 5.19.0 | Codificador de texto que interpreta el prompt              |
| `accelerate`   | >= 1.15.0 | Carga más rápida y eficiente de los pesos del modelo       |

## Uso

```bash
uv run python src/sd_turbo/__init__.py
```

El script pide el prompt por consola:

```
Cargando Modelo...
Escribe el promt de la imagen: un gato astronauta flotando en el espacio, estilo acuarela
Generando la imagen ...
Imagen guardada !
```

El resultado se guarda como `imagen.png` en la carpeta desde la que se ejecuta el script.

Además del prompt, el script usa un **negative prompt** fijo que le indica al modelo lo que debe evitar (baja calidad, desenfoque, anatomía deformada, texto, marcas de agua...). Se puede modificar en la variable `negative_prompt` de `__init__.py`.

## Parámetros de generación

La llamada `modelo(...)` acepta estos parámetros para ajustar la imagen:

| Parámetro             | Valor   | Descripción                                                        |
|-----------------------|---------|--------------------------------------------------------------------|
| `prompt`              | consola | Descripción de la imagen que se quiere generar                     |
| `negative_prompt`     | fijo    | Lo que el modelo debe evitar en la imagen                          |
| `num_inference_steps` | `10`    | Número de veces que crea y corrige la imagen                       |
| `guidance_scale`      | `2.0`   | Qué tan fiel será la imagen al prompt                              |
| `height` / `width`    | `1024`  | Alto y ancho de la imagen en píxeles                               |

Al cargar el modelo también se puede ajustar:

| Opción        | Valor           | Descripción                                                                 |
|---------------|-----------------|-----------------------------------------------------------------------------|
| `torch_dtype` | `torch.float32` | Precisión decimal del modelo. Si la imagen sale negra, prueba con `float16` |
| `.to(...)`    | `"cpu"`         | Dispositivo donde corre el modelo (ver [Notas](#notas))                     |

## Estructura del proyecto

```
sd-turbo/
├── src/
│   └── sd_turbo/
│       ├── __init__.py    # Script principal: carga el modelo y genera la imagen
│       └── imagen.png     # Imagen generada (salida de ejemplo)
├── pyproject.toml         # Configuración del proyecto y dependencias
├── uv.lock                # Versiones exactas de las dependencias
├── .python-version        # Versión de Python (3.12)
└── README.md
```

## Notas

- La primera ejecución tarda más porque descarga los pesos del modelo.
- Si tienes GPU, puedes cambiar `modelo.to("cpu")` en `__init__.py` para acelerar la generación:
  - `"cuda"`: tarjetas NVIDIA
  - `"mps"`: Mac con procesadores Apple Silicon (M1, M2...)
  - `"xpu"`: gráficas Intel
- SD-Turbo está entrenado para imágenes de **512 × 512** y para generar en muy pocos pasos (1 a 4), así que valores más bajos de `num_inference_steps` también dan buenos resultados y son más rápidos.

## Autor

samuelciapa972-creator
