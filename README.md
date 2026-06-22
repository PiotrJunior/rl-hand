Before first run:
```
uv sync
```
To run: 
```
uv run python src/hil.py --config_path src/env.json
uv run python src/learner.py --config_path src/train_config.json
uv run python src/actor.py --config_path src/train_config.json 
```