# Dataset Integration

This guide shows how to use datasets either downloaded from Roboflow or prepared locally, and how to feed the resulting `data.yaml` into the training and evaluation scripts.

## Roboflow

Requirements:
- Install: `pip install roboflow`
- Set API key via `ROBOFLOW_API_KEY` or pass `--api-key`

Download a YOLOv8-formatted dataset and get its `data.yaml` path:

```bash
python scripts/prepare_dataset.py \
  --source roboflow \
  --workspace lamar-university-venef \
  --project liquor-data \
  --version 4 \
  --format yolov8
```

This prints the absolute path to `data.yaml`, which you can pass to training scripts via `--data`.

Example:

```bash
DATA_YAML=$(python scripts/prepare_dataset.py --source roboflow \
  --workspace lamar-university-venef --project liquor-data --version 4)

python experiments/train_yolov8.py \
  --model yolov8n.pt \
  --data "$DATA_YAML" \
  --epochs 50 --imgsz 640 --batch 16
```

## Local Dataset

Ensure a directory structure:
```
/path/to/dataset/
  train/images/  train/labels/
  val/images/    val/labels/
  test/images/   test/labels/
```

If `data.yaml` exists in the dataset root, you can use it directly:
```bash
python experiments/train_yolov8.py --data /path/to/dataset/data.yaml ...
```

If `data.yaml` does not exist, provide a classes file to generate one:
- `.txt`: one class per line
- `.json`: list or `{ "names": [...] }`
- `.yaml`: list or `{ names: [...] }`

Generate `data.yaml`:
```bash
python scripts/prepare_dataset.py \
  --source local \
  --dataset-dir /path/to/dataset \
  --classes docs/classes.txt
```

This prints the path to the generated `data.yaml`.

## Notes
- Roboflow download location depends on the library; the script searches for `data.yaml` automatically.
- Store your API key securely: prefer `ROBOFLOW_API_KEY` in your environment.
- The integration is optional and does not affect CLI help tests or existing workflows.