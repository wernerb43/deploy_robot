##
#
# Download the latest policy checkpoint from a W&B run.
#
##

# standard library
import argparse
import re
from pathlib import Path

# for W&B API
import wandb

# this command works right now. 
#  python policy/get_policy_wandb.py sesteban-california-institute-of-technology-caltech/mjlab/bysdsnbu

def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("run_path", help="W&B run path (entity/project/run_id)")
  parser.add_argument(
    "--output-dir",
    default=".",
    help="Directory to save the checkpoint (default: current dir)",
  )
  parser.add_argument(
    "--checkpoint",
    default=None,
    help="Specific checkpoint name (e.g. model_4000.pt). Downloads latest if omitted.",
  )
  args = parser.parse_args()

  api = wandb.Api()
  run = api.run(args.run_path)
  all_files = [f.name for f in run.files()]
  pt_files = [f for f in all_files if re.match(r"^model_\d+\.pt$", f)]
  onnx_files = [f for f in all_files if f.endswith(".onnx")]

  print(f"Found {len(pt_files)} .pt files and {len(onnx_files)} .onnx files.")

  if args.checkpoint is not None:
    if args.checkpoint not in all_files:
      raise ValueError(f"Checkpoint '{args.checkpoint}' not found. Available: {pt_files + onnx_files}")
    checkpoint = args.checkpoint
  elif onnx_files:
    # prefer ONNX (self-contained, no training code needed)
    checkpoint = onnx_files[-1]
    print(f"Using ONNX file: {checkpoint}")
  elif pt_files:
    checkpoint = max(pt_files, key=lambda x: int(x.split("_")[1].split(".")[0]))
    print(f"Using latest .pt checkpoint: {checkpoint}")
    print("Note: .pt checkpoints require the training code to load. Consider exporting to ONNX.")
  else:
    raise RuntimeError(f"No model files found in run {args.run_path}")

  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  run.file(checkpoint).download(str(output_dir), replace=True)
  print(f"Downloaded {checkpoint} to {output_dir / checkpoint}")


if __name__ == "__main__":
  main()
