from pathlib import Path
import onnx
import logging
import subprocess
import tarfile
import sys
import datetime


# ===== Paths =====
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "PP_OCR_models"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_VALIDATE_DIR = BASE_DIR / "validation"
OUTPUT_VALIDATE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

LOG_FILE = OUTPUT_VALIDATE_DIR / "onnx_validation.log"

DET_TAR = MODEL_DIR / "ch_PP-OCRv4_det_infer.tar"
REC_TAR = MODEL_DIR / "ch_PP-OCRv4_rec_infer.tar"

DET_ONNX = OUTPUT_DIR / "det.onnx"
REC_ONNX = OUTPUT_DIR / "rec.onnx"

OPSET = 11


# ===== Logging Setup =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
# Shortcut for logging
log = logging.info


# ===== Run Cmd Function =====
def run_cmd(cmd, error_msg):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{error_msg}\n{result.stderr}")
    return result.stdout.strip()


# ===== Extract Tar Function =====
def extract_tar(tar_path: Path, dest: Path) -> Path:
    model_name = tar_path.stem
    extract_path = dest / model_name

    if extract_path.exists():
        log(f"Already extracted: {model_name}")
        return extract_path

    log(f"Extracting {tar_path.name}...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=dest)

    log(f"Extracted to {extract_path}")
    return extract_path

# ===== Paddle To Onnx Function =====
def paddle_to_onnx(paddle_dir: Path, onnx_out: Path):
    log(f"Converting {onnx_out.name} to ONNX...")
    cmd = [
        "paddle2onnx",
        "--model_dir", str(paddle_dir),
        "--model_filename", "inference.pdmodel",
        "--params_filename", "inference.pdiparams",
        "--save_file", str(onnx_out),
        "--opset_version", str(OPSET),
        "--enable_onnx_checker", "True",
    ]
    run_cmd(cmd, "Paddle to ONNX conversion failed")

    size = onnx_out.stat().st_size / (1024 ** 2)
    log(f"ONNX model saved: {onnx_out} ({size:.2f} MB)")


# ===== Validate Onnx Function =====
def validate_onnx(onnx_file: Path):
    log(f"Validating {onnx_file.name}...")
    try:
        model = onnx.load(str(onnx_file))
        onnx.checker.check_model(model)
        log(f"[OK] {onnx_file.name} is valid")

        # Input info
        log("Inputs:")
        for i in model.graph.input:
            log(f"  {i.name}: {i.type}")

        # Output info
        log("Outputs:")
        for o in model.graph.output:
            log(f"  {o.name}: {o.type}")

    except Exception as e:
        log(f"[ERROR] {onnx_file.name} failed validation: {e}")


# ===== Parent Function =====
def convert_models(tar_path: Path, output_onnx: Path, name: str):
    log(f"\n{'='*60}\nConverting {name} Model\n{'='*60}")
    paddle_dir = extract_tar(tar_path, MODEL_DIR)
    paddle_to_onnx(paddle_dir, output_onnx)
    validate_onnx(output_onnx)




# ===== Main =====
def main():
    log(f"PaddleOCR Model Converter (ONNX only) - {datetime.datetime.now()}\n")
    try:
        convert_models(DET_TAR, DET_ONNX, "Detection")
        convert_models(REC_TAR, REC_ONNX, "Recognition")
    except Exception as e:
        log(f"[ERROR] Conversion failed: {e}")
        sys.exit(1)

    log("\nConversion complete!")
    log(f"Output directory: {OUTPUT_DIR}")
    log(f"Validation log saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()

