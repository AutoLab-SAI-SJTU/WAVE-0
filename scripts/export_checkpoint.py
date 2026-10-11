#!/usr/bin/env python
"""Export a PI05-WAM checkpoint as portable WAVE-0 weights, not a training resume.

Only config, processors and their tensor payloads are exported. Tensor names and
data bytes stay unchanged; safetensors metadata is rebuilt without local paths.
The source is read-only and an existing destination is never overwritten.
"""

import argparse
import json
import re
import shutil
import struct
from pathlib import Path


def _portable_reference(value: str) -> str:
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", value) or value.startswith("."):
        raise ValueError("base_model must be a portable Hugging Face repo ID, not a local path or URL")
    return value


def _check_portable(value):
    if isinstance(value, dict):
        for item in value.values():
            _check_portable(item)
    elif isinstance(value, list):
        for item in value:
            _check_portable(item)
    elif isinstance(value, str) and (
        value.startswith(("/", "~", "file:", "http:", "https:", "hf_", "wandb_")) or "\\" in value
    ):
        raise ValueError("Non-portable string in exported metadata; inspect the source config locally")


def _copy_tensor_payload(source: Path, destination: Path, metadata: dict[str, str]):
    # Safetensors offsets are relative to the tensor-data section, not the file.
    # Rebuilding only the header preserves all tensor names, dtypes and bytes.
    with source.open("rb") as reader:
        size_bytes = reader.read(8)
        if len(size_bytes) != 8:
            raise ValueError(f"Invalid safetensors header: {source.name}")
        header_size = struct.unpack("<Q", size_bytes)[0]
        if header_size > 100_000_000:
            raise ValueError("Safetensors header exceeds the supported 100 MB limit")
        header = json.loads(reader.read(header_size))
        header["__metadata__"] = metadata
        encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
        encoded += b" " * (-len(encoded) % 8)
        with destination.open("xb") as writer:
            writer.write(struct.pack("<Q", len(encoded)))
            writer.write(encoded)
            shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)


def export_checkpoint(source: Path, destination: Path, base_model: str = "lerobot/pi05_base"):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Destination already exists; choose a new export directory")
    if source in destination.parents:
        raise ValueError("Export to a separate directory, not inside the source checkpoint")
    base_model = _portable_reference(base_model)
    config = json.loads((source / "config.json").read_text())
    if config.get("type") not in {"pi05_wam", "wave0"}:
        raise ValueError("Only original PI05-WAM / WAVE-0 checkpoints are supported")
    stage = config.get("world_model_stage")
    if stage not in {"stage1", "stage2", "inference"}:
        raise ValueError(f"Unsupported checkpoint stage: {stage!r}")
    config.update(
        type="wave0",
        world_model_stage="stage1" if stage == "stage1" else "inference",
        world_model_base_model_path=base_model,
        world_model_stage1_checkpoint_path=None,
        pretrained_path=None,
        repo_id=None,
        push_to_hub=False,
        device="cpu",
        compile_model=False,
    )
    # Publication tags may contain private experiment names. Recreate these in
    # the model card instead of copying historical metadata.
    config.pop("tags", None)
    _check_portable(config)
    json_files = {"config.json": config}
    tensor_files = {"model.safetensors"}
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        path = source / name
        if not path.is_file():
            if stage != "stage1":
                raise FileNotFoundError(f"Deployment export requires {name} and normalization state")
            continue
        pipeline = json.loads(path.read_text())
        # Keep only schema fields; don't copy arbitrary provenance.
        result = {"name": name.removesuffix(".json"), "steps": []}
        for step in pipeline["steps"]:
            exported = {
                key: step[key] for key in ("registry_name", "class", "config", "state_file") if key in step
            }
            for identifier in ("registry_name", "class"):
                if identifier in exported:
                    exported[identifier] = (
                        exported[identifier].replace("pi05_wam", "wave0").replace("Pi05WAM", "Wave0")
                    )
            if exported.get("registry_name") == "device_processor":
                exported["config"] = {**exported.get("config", {}), "device": "cpu"}
            state_file = exported.get("state_file")
            if state_file:
                if Path(state_file).name != state_file or not state_file.endswith(".safetensors"):
                    raise ValueError("Processor state must refer to a local .safetensors basename")
                tensor_files.add(state_file)
            result["steps"].append(exported)
        _check_portable(result)
        json_files[name] = result
    for name in tensor_files:
        if not (source / name).is_file():
            raise FileNotFoundError(f"Missing tensor file: {name}")

    # Exclusive creation also protects against accidentally exporting over a
    # checkpoint. On I/O failure, the partial directory remains for inspection.
    destination.mkdir(parents=True, exist_ok=False)
    for name in sorted(tensor_files):
        metadata = {"format": "pt"}
        if name == "model.safetensors" and stage == "stage1":
            metadata.update(wave0_checkpoint_type="stage1_world_only", base_model_path=base_model)
        _copy_tensor_payload(source / name, destination / name, metadata)
    for name, content in json_files.items():
        (destination / name).write_text(json.dumps(content, indent=2) + "\n")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Local pretrained_model directory")
    parser.add_argument("destination", type=Path, help="New, separate export directory")
    parser.add_argument("--base-model", default="lerobot/pi05_base")
    args = parser.parse_args()
    exported = export_checkpoint(args.source, args.destination, args.base_model)
    print(f"Exported WAVE-0 checkpoint to {exported}. Source unchanged; no optimizer/RNG/logs copied.")


if __name__ == "__main__":
    main()
