# custom

Download models from
[serengil/deepface_models](https://github.com/serengil/deepface_models) then put
in `~/.deepface/weights`.

Build:

```sh
uv venv
source .venv/bin/activate

uv pip install -r ./requirements.txt
uv pip install -e .

# Missing
uv pip install tf_keras
uv pip install facenet-pytorch
uv pip install humanize
```

Run:

```sh
source .venv/bin/activate

bash ./scripts/service.sh
```
