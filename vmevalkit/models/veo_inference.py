"""
Google Veo 3 (Vertex AI) — Text + Image → Video
- Auth: prefers Application Default Credentials (google-auth), falls back to `gcloud auth print-access-token`
- Models: veo-3.0-generate-001 (default) or veo-3.0-fast-generate-001
- Input image: local file (PNG/JPEG) or GCS URI
- Output: returns video bytes (if API responds inline) or downloads from GCS (if provided and client available)
"""

from __future__ import annotations

import os
import base64
import json
import time
import asyncio
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import httpx
from PIL import Image
import io

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv("LOGLEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)

# Attempt to load environment variables from a .env file if present
try:
    from dotenv import load_dotenv, find_dotenv
    _dotenv_path = find_dotenv(usecwd=True)
    if _dotenv_path:
        load_dotenv(_dotenv_path, override=False)
    else:
        # Fallback: try default search from CWD
        load_dotenv(override=False)
except Exception:
    # Safe fallback: proceed if python-dotenv is unavailable
    pass


def _hydrate_env_from_nearby_dotenv() -> None:
    """Best-effort manual .env loader if python-dotenv did not populate env.

    Searches for a .env starting from the current working directory and
    walking up a few parent directories relative to this file.
    Only sets variables that are not already present in os.environ.
    """
    candidate_paths = []
    try:
        candidate_paths.append(Path.cwd() / ".env")
    except Exception:
        pass
    try:
        here = Path(__file__).resolve()
        for i in range(1, 6):
            candidate_paths.append(here.parents[i] / ".env")
    except Exception:
        pass

    for env_path in candidate_paths:
        try:
            if env_path.exists():
                for raw_line in env_path.read_text().splitlines():
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
                break
        except Exception:
            continue

# -----------------------------
# Utilities: auth
# -----------------------------

def _google_access_token_via_adc() -> Optional[str]:
    """Try to obtain an access token via Application Default Credentials (google-auth)."""
    try:
        from google.auth import default
        from google.auth.transport.requests import Request

        creds, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        if not creds.valid:
            creds.refresh(Request())
        return creds.token
    except Exception as e:
        logger.debug(f"ADC auth not available: {e}")
        return None


def _google_access_token_via_gcloud() -> Optional[str]:
    """Fallback: obtain an access token from the gcloud CLI."""
    gcloud_paths = [
        shutil.which("gcloud"),
        "/usr/bin/gcloud",
        "/usr/local/bin/gcloud",
        "/home/ubuntu/google-cloud-sdk/bin/gcloud",
    ]
    gcloud_cmd = next((p for p in gcloud_paths if p and os.path.exists(p)), None)
    if not gcloud_cmd:
        logger.debug("gcloud not found on PATH or common locations.")
        return None
    try:
        token = subprocess.check_output([gcloud_cmd, "auth", "print-access-token"], text=True).strip()
        return token or None
    except subprocess.CalledProcessError as e:
        logger.debug(f"gcloud token retrieval failed: {e}")
        return None


def get_google_access_token() -> str:
    """Get an OAuth2 access token for Vertex AI."""
    token = _google_access_token_via_adc() or _google_access_token_via_gcloud()
    if not token:
        raise RuntimeError(
            "Could not obtain Google access token. "
            "Use Application Default Credentials (gcloud auth application-default login) "
            "or run `gcloud auth login` and keep gcloud available."
        )
    return token

# -----------------------------
# Veo 3 Client
# -----------------------------

class VeoService:
    """
    Vertex AI Veo 3 video generation client.
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        location: str = "us-central1",
        model_id: str = "veo-3.0-generate-preview",
        http_timeout_s: float = 600.0,
    ):
        # Accept multiple common env var names for GCP project and location
        self.project_id = (
            project_id
            or os.getenv("GOOGLE_PROJECT_ID")
            or os.getenv("GOOGLE_CLOUD_PROJECT")
            or os.getenv("GCP_PROJECT")
            or os.getenv("PROJECT_ID")
        )
        self.location = (
            os.getenv("GOOGLE_LOCATION")
            or os.getenv("GOOGLE_CLOUD_REGION")
            or os.getenv("REGION")
            or os.getenv("LOCATION")
            or location
        )
        self.model_id = model_id
        if not self.project_id:
            # As a last resort, parse a nearby .env and retry reading vars
            _hydrate_env_from_nearby_dotenv()
            self.project_id = (
                project_id
                or os.getenv("GOOGLE_PROJECT_ID")
                or os.getenv("GOOGLE_CLOUD_PROJECT")
                or os.getenv("GCP_PROJECT")
                or os.getenv("PROJECT_ID")
            )
            if not self.project_id:
                raise ValueError("GOOGLE_PROJECT_ID not set. Ensure it is in your .env or pass project_id=...")

        # Use regionless host per Vertex AI REST; region is in the path.
        self.base_url = "https://aiplatform.googleapis.com/v1"
        self.http_timeout_s = http_timeout_s

    # -------------------------
    # Public API
    # -------------------------

    async def generate_video(
        self,
        prompt: str,
        *,
        image_path: Optional[str] = None,
        image_gcs_uri: Optional[str] = None,
        duration_seconds: int = 8,          # allowed: 4, 6, 8
        aspect_ratio: str = "16:9",         # "16:9" or "9:16"
        resolution: str = "1080p",          # "720p" or "1080p"
        negative_prompt: Optional[str] = None,
        enhance_prompt: bool = True,
        generate_audio: bool = True,
        sample_count: int = 1,              # 1–4
        seed: Optional[int] = None,
        person_generation: Optional[str] = None,  # "disallow" | "allow_adult"
        storage_uri: Optional[str] = None,  # gs://bucket/prefix to write outputs
        poll_interval_s: float = 8.0,
        poll_timeout_s: float = 600.0,
        download_from_gcs: bool = False,    # if response points to GCS, try to download bytes
    ) -> Tuple[Optional[bytes], Dict[str, Any]]:
        """
        Submit a Veo 3 generation job and wait until completion.

        Returns:
            (video_bytes, metadata)
            - video_bytes: bytes of the first video if inline; or downloaded from GCS if requested; else None
            - metadata: full response dict (including any GCS URIs)
        """
        self._validate_params(duration_seconds, aspect_ratio, resolution, sample_count)

        token = get_google_access_token()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        instance: Dict[str, Any] = {"prompt": prompt}
        image_obj = self._build_image_object(
            image_path=image_path, 
            image_gcs_uri=image_gcs_uri,
            aspect_ratio=aspect_ratio
        )
        if image_obj:
            instance["image"] = image_obj

        parameters: Dict[str, Any] = {
            "durationSeconds": duration_seconds,
            "aspectRatio": aspect_ratio,
            "resolution": resolution,
            "generateAudio": generate_audio,
            "enhancePrompt": enhance_prompt,
            "sampleCount": sample_count,
        }
        if seed is not None:
            parameters["seed"] = int(seed)
        if negative_prompt:
            parameters["negativePrompt"] = negative_prompt
        if person_generation:
            parameters["personGeneration"] = person_generation
        if storage_uri:
            # Where Vertex AI should write outputs (video/audio) in GCS
            parameters["storageUri"] = storage_uri

        request_data = {"instances": [instance], "parameters": parameters}

        predict_url = (
            f"{self.base_url}/projects/{self.project_id}/locations/{self.location}"
            f"/publishers/google/models/{self.model_id}:predictLongRunning"
        )

        async with httpx.AsyncClient(timeout=self.http_timeout_s) as client:
            # Submit job
            resp = await client.post(predict_url, headers=headers, json=request_data)
            if resp.status_code != 200:
                self._raise_api_error("Submit", resp)

            op = resp.json()
            op_name = op.get("name")
            if not op_name:
                raise RuntimeError("No operation name returned from Veo 3 API.")

            logger.info(f"Veo 3 operation started: {op_name}")

            # Poll for completion
            response_data = await self._poll_operation(
                client=client,
                operation_name=op_name,
                headers=headers,
                poll_interval_s=poll_interval_s,
                poll_timeout_s=poll_timeout_s,
            )

        # Extract video
        videos = response_data.get("videos", [])
        if not videos:
            logger.warning("Operation completed but no videos found in response.")
            return None, response_data

        video0 = videos[0]
        # Inline bytes path
        if "bytesBase64Encoded" in video0:
            video_bytes = base64.b64decode(video0["bytesBase64Encoded"])
            logger.info(f"Received inline video bytes: {len(video_bytes)} bytes.")
            return video_bytes, response_data

        # GCS path
        if "gcsUri" in video0:
            gcs_uri = video0["gcsUri"]
            logger.info(f"Video available in GCS: {gcs_uri}")
            if download_from_gcs:
                maybe_bytes = self._download_from_gcs(gcs_uri)
                if maybe_bytes is not None:
                    logger.info(f"Downloaded {len(maybe_bytes)} bytes from GCS.")
                    return maybe_bytes, response_data
                else:
                    logger.warning("GCS download requested but google-cloud-storage not available or failed.")
            return None, response_data

        logger.warning("No inline bytes or GCS URI present for video[0].")
        return None, response_data

    async def save_video(self, video_bytes: bytes, output_path: Path) -> Path:
        """Save video bytes to a file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(video_bytes)
        logger.info(f"Video saved to {output_path}")
        return output_path

    # -------------------------
    # Internals
    # -------------------------

    def _validate_params(self, duration_seconds: int, aspect_ratio: str, resolution: str, sample_count: int) -> None:
        if duration_seconds not in (4, 6, 8):
            logger.warning(f"Veo 3 supports durations 4, 6, or 8 seconds; got {duration_seconds}. Using 8.")
            duration_seconds = 8
        if aspect_ratio not in ("16:9", "9:16"):
            raise ValueError('aspect_ratio must be "16:9" or "9:16"')
        if resolution not in ("720p", "1080p"):
            raise ValueError('resolution must be "720p" or "1080p"')
        if not (1 <= sample_count <= 4):
            raise ValueError("sample_count must be between 1 and 4")

    def _pad_image_to_aspect_ratio(self, image: Image.Image, target_aspect_ratio: str) -> Image.Image:
        """
        Pad an image with white margins to match the target aspect ratio.
        
        Args:
            image: PIL Image to pad
            target_aspect_ratio: "16:9" or "9:16"
            
        Returns:
            PIL Image with white padding to match target aspect ratio
        """
        # Parse target aspect ratio
        if target_aspect_ratio == "16:9":
            target_ratio = 16 / 9
        elif target_aspect_ratio == "9:16":
            target_ratio = 9 / 16
        else:
            raise ValueError(f"Unsupported aspect ratio: {target_aspect_ratio}")
        
        width, height = image.size
        current_ratio = width / height
        
        # If already correct ratio, return as-is
        if abs(current_ratio - target_ratio) < 0.01:
            return image
        
        # Calculate target dimensions
        if current_ratio > target_ratio:
            # Image is wider than target - add height padding
            new_width = width
            new_height = int(width / target_ratio)
        else:
            # Image is taller than target - add width padding
            new_height = height
            new_width = int(height * target_ratio)
        
        # Create white canvas with target dimensions
        padded = Image.new("RGB", (new_width, new_height), color="white")
        
        # Calculate position to center the original image
        x_offset = (new_width - width) // 2
        y_offset = (new_height - height) // 2
        
        # Paste original image onto white canvas
        padded.paste(image, (x_offset, y_offset))
        
        logger.info(f"Padded image from {width}×{height} to {new_width}×{new_height} for {target_aspect_ratio} ratio")
        return padded

    def _build_image_object(
        self,
        *,
        image_path: Optional[str],
        image_gcs_uri: Optional[str],
        aspect_ratio: str = "16:9",
    ) -> Optional[Dict[str, Any]]:
        """Create the 'image' object for an instance."""
        if image_path and image_gcs_uri:
            raise ValueError("Provide either image_path or image_gcs_uri, not both.")

        if image_path:
            p = Path(image_path)
            if not p.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Load image and pad to target aspect ratio
            image = Image.open(p)
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Pad image to match video aspect ratio (prevents cropping)
            padded_image = self._pad_image_to_aspect_ratio(image, aspect_ratio)
            
            # Convert to bytes
            buffer = io.BytesIO()
            padded_image.save(buffer, format="PNG")
            data = buffer.getvalue()
            
            return {
                "bytesBase64Encoded": base64.b64encode(data).decode("utf-8"),
                "mimeType": "image/png",  # Always PNG after processing
            }

        if image_gcs_uri:
            # e.g., "gs://my-bucket/path/to/frame.png"
            mime = "image/png" if image_gcs_uri.lower().endswith(".png") else "image/jpeg"
            return {
                "gcsUri": image_gcs_uri,
                "mimeType": mime,
            }

        return None

    async def _poll_operation(
        self,
        *,
        client: httpx.AsyncClient,
        operation_name: str,
        headers: Dict[str, str],
        poll_interval_s: float,
        poll_timeout_s: float,
    ) -> Dict[str, Any]:
        """Polls the Veo 3 long-running operation until completion."""
        poll_url = (
            f"{self.base_url}/projects/{self.project_id}/locations/{self.location}"
            f"/publishers/google/models/{self.model_id}:fetchPredictOperation"
        )

        deadline = time.time() + poll_timeout_s
        attempt = 0
        while True:
            if time.time() > deadline:
                raise TimeoutError(f"Operation timed out after {poll_timeout_s:.0f}s: {operation_name}")

            await asyncio.sleep(poll_interval_s)
            attempt += 1

            resp = await client.post(
                poll_url,
                headers=headers,
                json={"operationName": operation_name},
            )

            if resp.status_code != 200:
                logger.warning(f"[poll attempt {attempt}] non-200: {resp.status_code} {resp.text}")
                continue

            data = resp.json()
            done = data.get("done", False)
            if not done:
                logger.info(f"Operation still running... attempt={attempt}")
                continue

            if "error" in data:
                raise RuntimeError(f"Veo 3 operation failed: {json.dumps(data['error'])}")

            response_data = data.get("response", {})
            logger.info("Operation completed.")
            return response_data

    def _download_from_gcs(self, gcs_uri: str) -> Optional[bytes]:
        """
        Best-effort download from GCS using google-cloud-storage, if present.
        Returns bytes or None on failure/unavailable.
        """
        try:
            from google.cloud import storage
        except Exception:
            return None

        if not gcs_uri.startswith("gs://"):
            return None

        try:
            # Parse URI
            _, _, bucket_and_path = gcs_uri.partition("gs://")
            bucket_name, _, blob_path = bucket_and_path.partition("/")
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            return blob.download_as_bytes()
        except Exception as e:
            logger.warning(f"GCS download failed: {e}")
            return None

    @staticmethod
    def _raise_api_error(phase: str, resp: httpx.Response) -> None:
        try:
            payload = resp.json()
            msg = payload.get("error", {}).get("message", resp.text)
        except Exception:
            msg = resp.text
        raise RuntimeError(f"Veo 3 API {phase} error [{resp.status_code}]: {msg}")


# -----------------------------
# CLI Example
# -----------------------------
# Usage:
#   export GOOGLE_PROJECT_ID=your-project
#   export GOOGLE_LOCATION=us-central1
#   python -m vmevalkit.models.veo_inference --prompt "A gray mouse finds yellow cheese..." \
#       --image_path starter.png --out ./data/outputs/veo_output.mp4 --duration 8 --ratio 16:9 --res 1080p
#
# Or write to GCS (no inline bytes):
#   python -m vmevalkit.models.veo_inference --prompt "..." --image_path starter.png \
#       --storage_uri gs://your-bucket/veo-outputs/ --no-download
#
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Google Veo 3 — Text + Image → Video (Vertex AI)")
    parser.add_argument("--prompt", required=True, help="Text description for the video.")
    parser.add_argument("--image_path", default=None, help="Local PNG/JPEG to condition the video.")
    parser.add_argument("--image_gcs_uri", default=None, help="GCS URI to an input image (gs://bucket/path.png).")

    parser.add_argument("--duration", type=int, default=8, help="Duration seconds: 4, 6, or 8 (default: 8).")
    parser.add_argument("--ratio", default="16:9", choices=["16:9", "9:16"], help="Aspect ratio.")
    parser.add_argument("--res", default="1080p", choices=["720p", "1080p"], help="Resolution.")

    parser.add_argument("--negative_prompt", default=None, help="Things to avoid.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for repeatability.")
    parser.add_argument("--samples", type=int, default=1, help="Number of samples (1–4).")
    parser.add_argument("--no-enhance", dest="enhance", action="store_false", help="Disable prompt enhancement.")
    parser.add_argument("--no-audio", dest="audio", action="store_false", help="Disable native audio generation.")
    parser.add_argument("--person_generation", default=None, choices=["disallow", "allow_adult"], help="Person generation policy.")

    parser.add_argument("--storage_uri", default=None, help="gs://bucket/prefix for outputs (server-side).")
    parser.add_argument("--download", dest="download", action="store_true", help="If GCS is returned, try to download bytes.")
    parser.add_argument("--no-download", dest="download", action="store_false")
    parser.set_defaults(download=True)

    parser.add_argument("--model", default="veo-3.0-generate-preview",
                        choices=["veo-2.0-generate-001", "veo-3.0-generate-preview", "veo-3.0-fast-generate-preview"],
                        help="Veo model variant.")

    parser.add_argument("--out", default="./data/outputs/veo_output.mp4", help="Where to save video bytes if available.")
    args = parser.parse_args()

    async def _main():
        svc = VeoService(model_id=args.model)
        video_bytes, meta = await svc.generate_video(
            prompt=args.prompt,
            image_path=args.image_path,
            image_gcs_uri=args.image_gcs_uri,
            duration_seconds=args.duration,
            aspect_ratio=args.ratio,
            resolution=args.res,
            negative_prompt=args.negative_prompt,
            enhance_prompt=args.enhance,
            generate_audio=args.audio,
            sample_count=args.samples,
            seed=args.seed,
            person_generation=args.person_generation,
            storage_uri=args.storage_uri,
            download_from_gcs=args.download,
        )

        # Always write metadata for inspection
        meta_path = Path(args.out).with_suffix(".json")
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2))
        logger.info(f"Wrote operation response metadata to {meta_path}")

        if video_bytes:
            await svc.save_video(video_bytes, Path(args.out))
        else:
            logger.info(
                "No inline video bytes to save. "
                "If you provided --storage_uri, check your GCS bucket for outputs."
            )

    asyncio.run(_main())


