import argparse
import threading
import queue
import traceback
from typing import Any, Dict, Generator

import litserve as ls

from flashvsr_runner import run_flashvsr_integrated


class FlashVSRLitAPI(ls.LitAPI):
    """LitServe API for FlashVSR with streaming progress updates."""

    def setup(self, device: str) -> None:
        # LitServe will pass the resolved device string here (e.g., "cuda:0", "cpu").
        # We keep it for potential future use but default to the request's device="auto".
        self.device = device

    def decode_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        if "input_path" not in request:
            raise ValueError("'input_path' is required in the request payload.")

        cfg: Dict[str, Any] = {
            "input_path": request["input_path"],
            "model": request.get("model", "FlashVSR-v1.1"),
            "mode": request.get("mode", "tiny"),
            "scale": int(request.get("scale", 4)),
            "color_fix": bool(request.get("color_fix", True)),
            "tiled_vae": bool(request.get("tiled_vae", True)),
            "tiled_dit": bool(request.get("tiled_dit", False)),
            "tile_size": int(request.get("tile_size", 256)),
            "tile_overlap": int(request.get("tile_overlap", 24)),
            "unload_dit": bool(request.get("unload_dit", False)),
            "dtype": request.get("dtype", "bf16"),
            "seed": int(request.get("seed", -1)),
            "device": request.get("device", "auto"),
            "fps_override": int(request.get("fps_override", 30)),
            "quality": int(request.get("quality", 6)),
            "attention_mode": request.get("attention_mode", "sage"),
            "sparse_ratio": float(request.get("sparse_ratio", 2.0)),
            "kv_ratio": float(request.get("kv_ratio", 3.0)),
            "local_range": int(request.get("local_range", 11)),
        }
        return cfg

    def predict(self, cfg: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
        """Run FlashVSR and stream progress + final result via a generator."""

        events: "queue.Queue[Dict[str, Any]]" = queue.Queue()

        def progress_cb(value: float, desc: str = "") -> None:
            try:
                v = float(value)
            except Exception:
                v = 0.0
            events.put({"event": "progress", "progress": v, "desc": str(desc)})

        def worker() -> None:
            try:
                events.put({"event": "status", "message": "started"})
                output_path = run_flashvsr_integrated(
                    input_path=cfg["input_path"],
                    model=cfg["model"],
                    mode=cfg["mode"],
                    scale=cfg["scale"],
                    color_fix=cfg["color_fix"],
                    tiled_vae=cfg["tiled_vae"],
                    tiled_dit=cfg["tiled_dit"],
                    tile_size=cfg["tile_size"],
                    tile_overlap=cfg["tile_overlap"],
                    unload_dit=cfg["unload_dit"],
                    dtype_str=cfg["dtype"],
                    seed=cfg["seed"],
                    device=cfg["device"],
                    fps_override=cfg["fps_override"],
                    quality=cfg["quality"],
                    attention_mode=cfg["attention_mode"],
                    sparse_ratio=cfg["sparse_ratio"],
                    kv_ratio=cfg["kv_ratio"],
                    local_range=cfg["local_range"],
                    progress=progress_cb,
                )
                events.put(
                    {
                        "event": "result",
                        "progress": 1.0,
                        "output_path": output_path,
                    }
                )
            except Exception as e:
                tb = traceback.format_exc()
                events.put(
                    {
                        "event": "error",
                        "message": str(e),
                        "traceback": tb,
                    }
                )
            finally:
                events.put({"event": "end"})

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        while True:
            item = events.get()
            if item.get("event") == "end":
                break
            yield item

        thread.join()

    def encode_response(self, output: Generator[Dict[str, Any], None, None]):
        for item in output:
            yield item


def main() -> None:
    parser = argparse.ArgumentParser(description="FlashVSR LitServe server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument(
        "--timeout",
        type=float,
        default=-1,
        help="Per-request timeout in seconds (-1 or <0 to disable)",
    )
    args = parser.parse_args()

    api = FlashVSRLitAPI()
    server = ls.LitServer(
        api,
        accelerator="auto",
        stream=True,
        timeout=args.timeout,
    )
    server.run(host=args.host, port=args.port, generate_client_file=False)


if __name__ == "__main__":
    main()
