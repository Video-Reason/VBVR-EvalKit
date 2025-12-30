import os
import time
import asyncio
import logging
from typing import Optional, Dict, Any, Union, Tuple
from pathlib import Path
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from .base import ModelWrapper

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

try:
    import httpx
except ImportError as e:
    raise ImportError("Please `pip install httpx pillow`") from e


class SoraService:
    """
    Service for image-to-video generation using OpenAI Sora models.
    """

    def __init__(
        self,
        model: str = "sora-2",
        base_url: str = "https://api.openai.com/v1",
        idempotency_prefix: Optional[str] = None,
        request_timeout_sec: float = 300.0,
    ):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.idempotency_prefix = idempotency_prefix
        self.request_timeout_sec = request_timeout_sec

        # Constraints aligned; do not silently coerce outside these.
        self.model_constraints = self._get_model_constraints(model)

        # Upload field name kept configurable (default matches cookbook usage).
        self.upload_field_name = "input_reference"

        # Allowed upload MIME types
        self.allowed_mimes = {"image/jpeg", "image/png", "image/webp"}

    def _get_model_constraints(self, model: str) -> Dict[str, Any]:
        constraints = {
            "sora-2": {
                "durations": ["4", "8", "12"],  # send as strings in form fields
                "sizes": ["1280x720", "720x1280"],
                "description": "OpenAI Sora-2 - High-quality video generation",
            },
            "sora-2-pro": {
                "durations": ["4", "8", "12"],
                "sizes": ["1280x720", "720x1280", "1024x1792", "1792x1024"],
                "description": "OpenAI Sora-2-Pro - Enhanced model with more resolution options",
            },
        }
        if model not in constraints:
            raise ValueError(f"Unknown Sora model: {model}. Available: {list(constraints.keys())}")
        return constraints[model]

    # ---------- Public API ----------

    async def generate_video(
        self,
        prompt: str,
        image_path: Union[str, Path],
        duration: Union[int, str] = "8",
        size: str = "1280x720",
        output_path: Optional[Union[str, Path]] = None,
        auto_pad: bool = True,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate video from text + image.
        - duration must be one of ["4","8","12"] (int or str accepted; sent as str)
        - size must exactly match one of the model's supported sizes
        """

        # Normalize/validate duration (store/send as string)
        duration_str = str(duration)
        if duration_str not in self.model_constraints["durations"]:
            raise ValueError(
                f"Duration {duration} not supported for {self.model}. "
                f"Allowed: {self.model_constraints['durations']}"
            )

        # Validate size strictly (no silent coercion)
        if size not in self.model_constraints["sizes"]:
            raise ValueError(
                f"Size {size} not supported for {self.model}. "
                f"Allowed: {self.model_constraints['sizes']}"
            )

        # Prepare upload image (exact target size; convert/resize/pad in-memory)
        filename, file_bytes, mime = await self._prepare_image_for_upload(image_path, size, auto_pad)

        # Create video job
        video_id = await self._create_video_job(
            prompt=prompt,
            image_file=(filename, file_bytes, mime),
            duration_str=duration_str,
            size=size,
            idempotency_key=idempotency_key,
        )
        logger.info(f"Sora video generation started. Job ID: {video_id}")

        # Poll to terminal status (with 99% grace)
        job_result = await self._poll_video_job(video_id)

        status = job_result.get("status")
        if status not in {"completed", "succeeded"}:
            # Bubble up clear error with any payload
            raise Exception(f"Video generation ended without success: {job_result}")

        result = {
            "video_id": video_id,
            "model": self.model,
            "prompt": prompt,
            "image_path": str(image_path),
            "duration": duration_str,
            "size": size,
            "status": status,
        }

        # Download
        if output_path:
            if not isinstance(output_path, Path):
                output_path = Path(output_path)
            saved_path = await self._download_video(video_id, output_path)
            result["video_path"] = str(saved_path)
            logger.info(f"Video saved to: {saved_path}")

        return result

    # ---------- Image prep ----------

    async def _prepare_image_for_upload(
        self,
        image_path: Union[str, Path],
        target_size: str,
        auto_pad: bool,
    ) -> Tuple[str, BytesIO, str]:
        """
        Loads, converts (jpeg/png/webp), resizes/pads in memory to exact target_size.
        Returns (filename, bytes_io, mime)
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        target_w, target_h = map(int, target_size.split("x"))

        with Image.open(path) as img:
            # Normalize color and orientation
            img = img.convert("RGB")
            current_w, current_h = img.size

            if (current_w, current_h) != (target_w, target_h):
                if not auto_pad:
                    raise ValueError(
                        f"Image resolution {current_w}x{current_h} does not match target {target_size}. "
                        "Set auto_pad=True to pad/fit automatically."
                    )
                # Pad with neutral gray to avoid harsh borders
                padded = Image.new("RGB", (target_w, target_h), color=(128, 128, 128))
                scale = min(target_w / current_w, target_h / current_h)
                new_w = max(1, int(current_w * scale))
                new_h = max(1, int(current_h * scale))
                resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                x_offset = (target_w - new_w) // 2
                y_offset = (target_h - new_h) // 2
                padded.paste(resized, (x_offset, y_offset))
                img = padded

        # Choose a safe default format (PNG) unless the original is clearly JPEG/WEBP
        ext = path.suffix.lower()
        if ext in {".jpg", ".jpeg"}:
            mime = "image/jpeg"
            pil_format = "JPEG"
            filename = path.stem + ".jpg"
        elif ext == ".webp":
            mime = "image/webp"
            pil_format = "WEBP"
            filename = path.stem + ".webp"
        else:
            mime = "image/png"
            pil_format = "PNG"
            filename = path.stem + ".png"

        if mime not in self.allowed_mimes:
            # Force to PNG if something odd
            mime = "image/png"
            pil_format = "PNG"
            filename = path.stem + ".png"

        bio = BytesIO()
        img.save(bio, format=pil_format)
        bio.seek(0)
        return filename, bio, mime

    # ---------- HTTP helpers ----------

    def _auth_headers(self, idempotency_key: Optional[str] = None) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return headers

    async def _request_with_retries(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        max_attempts: int = 5,
        backoff_base: float = 0.5,
        **kwargs,
    ) -> httpx.Response:
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = await client.request(method, url, timeout=self.request_timeout_sec, **kwargs)
                if resp.status_code not in (429, 500, 502, 503, 504):
                    return resp
                logger.warning(f"{method} {url} -> {resp.status_code}; retrying...")
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                logger.warning(f"{method} {url} network/timeout error: {e}; retrying...")

            if attempt >= max_attempts:
                # On last attempt, just return (or re-raise)
                return resp

            sleep_s = backoff_base * (2 ** (attempt - 1))
            await asyncio.sleep(sleep_s)

    # ---------- Core API calls ----------

    async def _create_video_job(
        self,
        prompt: str,
        image_file: Tuple[str, BytesIO, str],
        duration_str: str,
        size: str,
        idempotency_key: Optional[str],
    ) -> str:
        url = f"{self.base_url}/videos"

        headers = self._auth_headers(
            idempotency_key=idempotency_key or (self.idempotency_prefix or "sora") + f"-{int(time.time()*1000)}"
        )

        filename, file_bytes, mime = image_file

        data = {
            "model": self.model,
            "prompt": prompt,
            "size": size,           # exact match to constraints
            "seconds": duration_str # send as string for form fields
        }

        files = {
            self.upload_field_name: (filename, file_bytes, mime)
        }

        async with httpx.AsyncClient() as client:
            resp = await self._request_with_retries(
                client, "POST", url, headers=headers, data=data, files=files
            )

        if resp.status_code != 200:
            raise Exception(f"Failed to create video job: {resp.status_code} {resp.text}")

        job = resp.json()
        vid = job.get("id")
        if not vid:
            raise Exception(f"Invalid create response (no id): {job}")
        return vid

    async def _poll_video_job(self, video_id: str, max_wait_time: int = 14400) -> Dict[str, Any]:
        """
        Poll until terminal status. Adds a grace window if stuck at 99% post-processing.
        """
        url = f"{self.base_url}/videos/{video_id}"
        headers = self._auth_headers()
        terminal = {"completed", "succeeded", "failed", "cancelled", "rejected"}

        start = time.time()
        last_status = None
        interval = 2.0

        async with httpx.AsyncClient() as client:
            while time.time() - start < max_wait_time:
                resp = await self._request_with_retries(client, "GET", url, headers=headers)
                if resp.status_code != 200:
                    logger.warning(f"Poll {video_id} -> {resp.status_code}; continuing")
                    await asyncio.sleep(min(10.0, interval))
                    interval = min(10.0, interval * 1.5)
                    continue

                job = resp.json()
                status = job.get("status", "unknown")
                progress = job.get("progress")

                if status != last_status:
                    logger.info(f"[{video_id}] status={status} progress={progress}")
                    last_status = status

                # Terminal handling
                if status in terminal:
                    if status not in {"completed", "succeeded"}:
                        # surface error details if present
                        err = job.get("error")
                        raise Exception(f"Video generation failed: status={status} error={err or job}")
                    return job

                # 99% grace window for post-processing
                if progress == 99 and status == "in_progress":
                    grace_start = time.time()
                    while time.time() - grace_start < 900:  # 15 minutes
                        await asyncio.sleep(3)
                        r = await self._request_with_retries(client, "GET", url, headers=headers)
                        if r.status_code != 200:
                            continue
                        j2 = r.json()
                        s2 = j2.get("status")
                        if s2 in terminal:
                            if s2 not in {"completed", "succeeded"}:
                                raise Exception(f"Video generation failed after 99%: {j2.get('error') or j2}")
                            return j2
                    raise TimeoutError("Video stuck at 99% post-processing beyond grace window.")

                await asyncio.sleep(interval)
                interval = min(10.0, interval * 1.25)

        raise TimeoutError(f"Video generation timed out after {max_wait_time} seconds")

    async def _download_video(self, video_id: str, output_path: Path) -> Path:
        if not isinstance(output_path, Path):
            output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        url = f"{self.base_url}/videos/{video_id}/content"
        headers = self._auth_headers()

        async with httpx.AsyncClient() as client:
            async with client.stream("GET", url, headers=headers, timeout=self.request_timeout_sec) as resp:
                if resp.status_code != 200:
                    raise Exception(f"Failed to download video: {resp.status_code} {await resp.aread()}")

                with open(output_path, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

        return output_path


# ========================================
# VMEVALKIT WRAPPER CLASS
# ========================================

class OpenAIWrapper(ModelWrapper):
    """
    VMEvalKit wrapper for SoraService to match the standard interface.
    """
    
    def __init__(
        self,
        model: str,
        output_dir: str = "./data/outputs",
        **kwargs
    ):
        """Initialize OpenAI Sora wrapper."""
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.kwargs = kwargs
        
        # Create SoraService instance
        self.sora_service = SoraService(model=model)
    
    def generate(
        self,
        image_path: Union[str, Path],
        text_prompt: str,
        duration: float = 8.0,
        output_filename: Optional[str] = None,
        size: str = "1280x720",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate video using Sora (synchronous wrapper).
        
        Args:
            image_path: Path to input image (must match size exactly)
            text_prompt: Text prompt for video generation
            duration: Video duration in seconds (4, 8, or 12)
            output_filename: Optional output filename
            size: Video resolution (must match image dimensions exactly)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with generation results
        """
        import time
        start_time = time.time()
        
        # Convert duration to int (Sora expects int)
        duration_int = int(duration)
        
        # Generate output path
        if not output_filename:
            # Create filename from model and timestamp
            safe_model = self.model.replace('-', '_')
            timestamp = int(time.time())
            output_filename = f"sora_{safe_model}_{timestamp}.mp4"
        
        output_path = self.output_dir / output_filename
        
        # Run async generation in sync context
        # Filter kwargs - SoraService.generate_video only accepts:
        # prompt, image_path, duration, size, output_path, auto_pad
        # All parameters are already passed explicitly, so no additional kwargs
        result = asyncio.run(
            self.sora_service.generate_video(
                prompt=text_prompt,
                image_path=str(image_path),
                duration=duration_int,
                size=size,
                output_path=output_path,
                auto_pad=True  # Enable auto-padding by default
            )
        )
        
        duration_taken = time.time() - start_time
        
        return {
            "success": bool(result.get("video_path")),
            "video_path": result.get("video_path"),
            "error": None,
            "duration_seconds": duration_taken,
            "generation_id": result.get("video_id", 'unknown'),
            "model": self.model,
            "status": "success" if result.get("video_path") else "failed",
            "metadata": {
                "prompt": text_prompt,
                "image_path": str(image_path),
                "duration": duration_int,
                "size": result.get("size"),
                "sora_result": result
            }
        }
