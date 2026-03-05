from __future__ import annotations

import logging
import os
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_QWEN_DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


class VideoLLM:
    """
    Qwen2.5-Omni video+audio understanding backend.

    Loads the model once at construction and keeps it in GPU memory.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Omni-7B",
        torch_dtype: str = "auto",
        device_map: str = "auto",
        max_frames: int = 64,
        total_pixels: int = 20_971_520,
        use_audio: bool = False,
    ) -> None:
        from transformers import (  # type: ignore
            Qwen2_5OmniForConditionalGeneration,
            Qwen2_5OmniProcessor,
        )

        logger.info("Loading VideoLLM: %s", model_name)

        # Use flash_attention_2 for the vision encoder if flash_attn is installed.
        # Transformers propagates attn_implementation to the thinker but NOT to
        # thinker.vision_config, so we patch it after config loading.
        _attn_impl = "flash_attention_2"
        try:
            import flash_attn as _fa  # noqa: F401
            logger.info("flash_attn %s found – using flash_attention_2", _fa.__version__)
        except ImportError:
            _attn_impl = None
            logger.info("flash_attn not found – using default attention")

        load_kwargs: dict = dict(
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        if _attn_impl:
            from transformers import AutoConfig  # type: ignore
            cfg = AutoConfig.from_pretrained(model_name)
            # Force vision encoder inside thinker to use flash attention.
            if hasattr(cfg, "thinker_config") and hasattr(cfg.thinker_config, "vision_config"):
                cfg.thinker_config.vision_config._attn_implementation = _attn_impl
            load_kwargs["config"] = cfg
            load_kwargs["attn_implementation"] = _attn_impl

        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_name,
            **load_kwargs,
        )
        # We only need text output – free the talker (audio generation) VRAM.
        self.model.disable_talker()

        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
        self.model_name = model_name
        self.max_frames = max_frames
        self.total_pixels = total_pixels
        self.use_audio = use_audio
        logger.info("VideoLLM ready (max_frames=%d, total_pixels=%d, audio=%s)",
                     max_frames, total_pixels, use_audio)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(
        self,
        question: str,
        video_path: str,
        skill_context: Optional[str] = None,
        use_audio_in_video: Optional[bool] = None,
    ) -> str:
        if use_audio_in_video is None:
            use_audio_in_video = self.use_audio
        try:
            return self._generate(question, video_path, skill_context, use_audio_in_video)
        except Exception:
            if use_audio_in_video:
                logger.warning("Audio decoding failed; retrying without audio.")
                return self._generate(question, video_path, skill_context, False)
            raise

    def answer_images(
        self,
        question: str,
        images: list,
    ) -> str:
        """Answer using high-resolution still frames (no video)."""
        return self._generate_images(question, images)

    def answer_with_crops(
        self,
        question: str,
        video_path: str,
        crops_b64: list[str],
        skill_context: Optional[str] = None,
        target_desc: str = "target region",
    ) -> str:
        """Answer with video (temporal context) + high-res crops (spatial detail).

        This is the core of the "Spatio-Temporal Attention" paradigm:
        - The video gives the model full temporal understanding at low resolution.
        - The crops give pixel-perfect detail of the region of interest.
        No text conversion — raw pixels flow directly into the VLM.

        Falls back to text-only answer() on OOM or decoding error.
        """
        try:
            return self._generate_with_crops(
                question, video_path, crops_b64, skill_context, target_desc
            )
        except Exception as exc:
            logger.warning(
                "[VideoLLM.answer_with_crops] Failed (%s); falling back to video-only.", exc
            )
            return self.answer(question, video_path, skill_context)

    # ------------------------------------------------------------------
    # Internal generation
    # ------------------------------------------------------------------

    def _generate(
        self,
        question: str,
        video_path: str,
        skill_context: Optional[str] = None,
        use_audio_in_video: bool = True,
    ) -> str:
        # -- build conversation --
        user_content: list[dict] = [{
            "type": "video",
            "video": video_path,
            "max_frames": self.max_frames,
            "total_pixels": self.total_pixels,
        }]

        text_parts = [question]
        if skill_context:
            text_parts.append(
                "\n\nAdditional context from specialised analysis tools:\n"
                f"{skill_context}"
                "\n\nBefore answering, you MUST provide a step-by-step reasoning"
                " process enclosed in <reasoning> tags."
                " In your reasoning, perform a Cross-Modal Consistency Check:\n"
                "1. What do you directly observe in the video?\n"
                "2. What does the tool evidence say?\n"
                "3. Are there any conflicts? (e.g., the ASR talks about X, but the"
                " video visually shows Y at the relevant moment).\n"
                "4. If the question asks about a VISUAL event or something seen on"
                " screen, trust your direct video observation over ASR text — ASR"
                " captures speech across the whole video and may reflect earlier"
                " content, not the specific moment the question targets.\n"
                "After </reasoning>, output ONLY the final answer letter (A, B, C, or D)."
            )
        user_content.append({"type": "text", "text": "\n".join(text_parts)})

        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": os.getenv("VIDEO_LLM_SYSTEM_PROMPT", _QWEN_DEFAULT_SYSTEM_PROMPT),
                    }
                ],
            },
            {"role": "user", "content": user_content},
        ]

        return self._run_model(conversation, use_audio_in_video)

    def _generate_with_crops(
        self,
        question: str,
        video_path: str,
        crops_b64: list[str],
        skill_context: Optional[str],
        target_desc: str,
    ) -> str:
        """Build a multi-modal conversation: video + high-res crops + question."""
        import base64 as _b64
        from PIL import Image as PILImage

        user_content: list[dict] = [
            {
                "type": "video",
                "video": video_path,
                "max_frames": self.max_frames,
                "total_pixels": self.total_pixels,
            }
        ]

        # Decode base64 crops back to PIL and inject as high-res images.
        decoded_crops: list[PILImage.Image] = []
        for b64_str in crops_b64:
            raw = _b64.b64decode(b64_str)
            import io as _io
            pil = PILImage.open(_io.BytesIO(raw)).convert("RGB")
            decoded_crops.append(pil)
            user_content.append({"type": "image", "image": pil})

        # Compose the prompt — tell the model what the crops show.
        crop_note = (
            f"\n\n[VISUAL DETAIL] The {len(decoded_crops)} image(s) above are "
            f"high-resolution crops from the video, zoomed in on: '{target_desc}'. "
            "Use these crops for pixel-level detail that may not be visible in the "
            "lower-resolution video stream (e.g., small text, logos, fine-grained "
            "object attributes)."
        )
        text_parts = [question, crop_note]
        if skill_context:
            text_parts.append(
                f"\n\nAdditional context from other analysis tools:\n{skill_context}"
            )
        text_parts.append(
            "\n\nReason step-by-step in <reasoning> tags."
            "\nAfter </reasoning>, output ONLY the final answer letter (A, B, C, or D)."
        )
        user_content.append({"type": "text", "text": "\n".join(text_parts)})

        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": os.getenv(
                    "VIDEO_LLM_SYSTEM_PROMPT", _QWEN_DEFAULT_SYSTEM_PROMPT
                )}],
            },
            {"role": "user", "content": user_content},
        ]

        logger.info(
            "[VideoLLM] answer_with_crops: video=%s  crops=%d  target='%s'",
            video_path, len(decoded_crops), target_desc,
        )
        return self._run_model(conversation, use_audio_in_video=False)

    def _generate_images(self, question: str, images: list) -> str:
        """Generate answer from high-res still images (no video)."""
        from PIL import Image as PILImage

        user_content: list[dict] = []
        for img in images:
            if isinstance(img, PILImage.Image):
                user_content.append({"type": "image", "image": img})
            else:
                user_content.append({"type": "image", "image": str(img)})
        user_content.append({"type": "text", "text": question})

        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": _QWEN_DEFAULT_SYSTEM_PROMPT}],
            },
            {"role": "user", "content": user_content},
        ]

        return self._run_model(conversation, use_audio_in_video=False)

    def _run_model(
        self,
        conversation: list[dict],
        use_audio_in_video: bool = False,
    ) -> str:
        """Shared generation logic for both video and image inputs."""
        from qwen_omni_utils import process_mm_info  # type: ignore

        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(
            conversation, use_audio_in_video=use_audio_in_video
        )
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_audio_in_video,
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        text_ids = self.model.generate(
            **inputs,
            use_audio_in_video=use_audio_in_video,
            return_audio=False,
            max_new_tokens=512,
            repetition_penalty=1.05,
        )

        decoded = self.processor.batch_decode(
            text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        full_text = decoded[0] if decoded else ""

        # Free GPU memory from intermediate tensors.
        del inputs, audios, images, videos, text_ids, conversation
        import gc
        gc.collect()
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass

        marker = "assistant\n"
        if marker in full_text:
            full_text = full_text.rsplit(marker, 1)[-1]
        return full_text.strip()

    # ------------------------------------------------------------------
    # Frame extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_key_frames(
        video_path: str,
        n_frames: int = 5,
        max_long_edge: int = 1280,
    ) -> list:
        """Extract evenly-spaced frames as PIL Images at high resolution.

        Returns up to *n_frames* PIL.Image.Image objects.
        Each frame is resized so the longest edge is at most *max_long_edge*.
        """
        from PIL import Image as PILImage

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return []

        # Pick evenly-spaced frame indices (avoid first/last 2% for slate frames).
        start_idx = max(0, int(total * 0.02))
        end_idx = min(total - 1, int(total * 0.98))
        if end_idx <= start_idx:
            start_idx, end_idx = 0, total - 1
        indices = np.linspace(start_idx, end_idx, n_frames, dtype=int)

        frames: list = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, bgr = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            long_edge = max(h, w)
            if long_edge > max_long_edge:
                scale = max_long_edge / long_edge
                rgb = cv2.resize(
                    rgb, (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            frames.append(PILImage.fromarray(rgb))
        cap.release()
        return frames
