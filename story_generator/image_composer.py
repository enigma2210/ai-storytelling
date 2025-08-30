from PIL import Image, ImageFilter, ImageEnhance, ImageChops, ImageDraw
import os
import logging
import uuid
from django.conf import settings
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class ImageComposer:
    """
    Advanced image composer that creates true color blending - like mixing paint.
    No superimposition - creates genuine color amalgamation.
    """

    def __init__(self):
        self.output_size = (800, 600)

    def compose_scene(self, background_path: str, character_path: str) -> Optional[str]:
        """
        Create true color blending like mixing red + white = pink.
        No layering or superimposition - genuine color fusion.
        """
        try:
            if not background_path or not character_path:
                logger.warning("Missing image paths for composition")
                return None

            # Build full paths
            bg_full_path = os.path.join(settings.MEDIA_ROOT, background_path)
            char_full_path = os.path.join(settings.MEDIA_ROOT, character_path)

            if not os.path.exists(bg_full_path) or not os.path.exists(char_full_path):
                logger.error("Image files not found for composition")
                return None

            # Load images
            background = Image.open(bg_full_path).convert('RGB')
            character = Image.open(char_full_path).convert('RGB')

            logger.info(f"Creating true color blend - Background: {background.size}, Character: {character.size}")

            # Create true color mixing
            blended_scene = self._create_true_color_blend(background, character)

            if not blended_scene:
                logger.error("Failed to create color blend")
                return None

            # Save the blended scene
            output_path = self._save_composed_image(blended_scene)

            logger.info(f"Successfully created color-blended scene: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Scene composition error: {e}")
            return None

    def _create_true_color_blend(self, background: Image.Image, character: Image.Image) -> Optional[Image.Image]:
        """
        Create true color blending using mathematical color mixing.
        Like mixing paints - not layering them.
        """
        try:
            # Standardize background size
            background = background.resize(self.output_size, Image.Resampling.LANCZOS)

            # Convert to numpy arrays for precise color manipulation
            bg_array = np.array(background).astype(np.float32)
            char_array = np.array(character).astype(np.float32)

            # Resize character to blend area size
            char_width = int(self.output_size[0] * 0.4)
            char_height = int(character.size[1] * (char_width / character.size[0]))

            # Limit character height
            max_height = int(self.output_size[1] * 0.65)
            if char_height > max_height:
                char_height = max_height
                char_width = int(character.size[0] * (char_height / character.size[1]))

            # Resize character using PIL then convert to numpy
            character_resized = character.resize((char_width, char_height), Image.Resampling.LANCZOS)
            char_array = np.array(character_resized).astype(np.float32)

            # Position for blending (center-left, bottom)
            blend_x = 60
            blend_y = self.output_size[1] - char_height - 50

            # Create true color blending in the specified region
            blended_array = self._apply_color_mixing(bg_array, char_array, blend_x, blend_y)

            # Convert back to PIL image
            blended_image = Image.fromarray(blended_array.astype(np.uint8))

            # Apply final artistic enhancement
            final_image = self._apply_artistic_enhancement(blended_image)

            return final_image

        except Exception as e:
            logger.error(f"True color blend creation error: {e}")
            return None

    def _apply_color_mixing(self, bg_array: np.ndarray, char_array: np.ndarray, x: int, y: int) -> np.ndarray:
        """
        Apply true mathematical color mixing - not alpha blending.
        This creates the red + white = pink effect.
        """
        try:
            result_array = bg_array.copy()
            char_h, char_w = char_array.shape[:2]

            # Create blend mask with smooth gradients
            blend_mask = self._create_blend_mask(char_w, char_h)

            # Extract the region to blend
            bg_region = bg_array[y:y + char_h, x:x + char_w]

            # Apply different blending modes for natural look
            for i in range(char_h):
                for j in range(char_w):
                    if i < bg_region.shape[0] and j < bg_region.shape[1]:
                        blend_factor = blend_mask[i, j]

                        if blend_factor > 0.1:  # Only blend where there's significant contribution
                            # Get colors
                            bg_color = bg_region[i, j]
                            char_color = char_array[i, j]

                            # True color mixing using different blend modes
                            mixed_color = self._mix_colors(bg_color, char_color, blend_factor)

                            # Apply the mixed color
                            result_array[y + i, x + j] = mixed_color

            return result_array

        except Exception as e:
            logger.error(f"Color mixing application error: {e}")
            return bg_array

    def _create_blend_mask(self, width: int, height: int) -> np.ndarray:
        """
        Create a sophisticated blend mask for natural color mixing.
        """
        # Create base mask
        mask = np.ones((height, width), dtype=np.float32)

        # Create radial falloff from center
        center_x, center_y = width // 2, height // 2

        for i in range(height):
            for j in range(width):
                # Distance from center
                dist_x = (j - center_x) / (width / 2)
                dist_y = (i - center_y) / (height / 2)
                distance = np.sqrt(dist_x ** 2 + dist_y ** 2)

                # Create smooth falloff
                if distance <= 1.0:
                    mask[i, j] = 1.0 - (distance * 0.6)  # Softer falloff
                else:
                    mask[i, j] = 0.4 * (1.0 / distance)  # Gentle outer blend

        # Apply Gaussian smoothing for very natural transitions
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=12))
        smooth_mask = np.array(mask_img).astype(np.float32) / 255.0

        return smooth_mask

    def _mix_colors(self, bg_color: np.ndarray, char_color: np.ndarray, blend_factor: float) -> np.ndarray:
        """
        Mix colors using various blending modes for natural results.
        This is where the magic happens - true color fusion!
        """
        # Normalize blend factor
        blend_factor = max(0.0, min(1.0, blend_factor))

        # Use different mixing strategies
        if blend_factor > 0.7:  # Strong blend - use color mixing
            # Multiplicative blending for rich colors
            mixed = (bg_color * char_color / 255.0)
            # Add some additive blending
            mixed = 0.6 * mixed + 0.4 * ((bg_color + char_color) / 2)
        elif blend_factor > 0.4:  # Medium blend - use soft light
            # Soft light blending
            mixed = np.zeros_like(bg_color)
            for c in range(3):
                if char_color[c] < 128:
                    mixed[c] = bg_color[c] * char_color[c] / 128
                else:
                    mixed[c] = 255 - (255 - bg_color[c]) * (255 - char_color[c]) / 128
        else:  # Light blend - use overlay
            # Overlay blending
            mixed = np.zeros_like(bg_color)
            for c in range(3):
                if bg_color[c] < 128:
                    mixed[c] = 2 * bg_color[c] * char_color[c] / 255
                else:
                    mixed[c] = 255 - 2 * (255 - bg_color[c]) * (255 - char_color[c]) / 255

        # Final interpolation with original background
        final_color = (1 - blend_factor) * bg_color + blend_factor * mixed

        return np.clip(final_color, 0, 255)

    def _apply_artistic_enhancement(self, image: Image.Image) -> Image.Image:
        """
        Apply artistic enhancements to make the blend look natural and professional.
        """
        try:
            # Very subtle overall blur for cohesion
            enhanced = image.filter(ImageFilter.GaussianBlur(radius=0.8))

            # Color harmony adjustment
            enhancer = ImageEnhance.Color(enhanced)
            enhanced = enhancer.enhance(1.08)

            # Contrast for definition but not harsh
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(1.03)

            # Slight brightness adjustment
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(1.01)

            # Apply unsharp mask for subtle detail enhancement
            blurred = enhanced.filter(ImageFilter.GaussianBlur(radius=1))
            enhanced = ImageChops.add(enhanced, ImageChops.subtract(enhanced, blurred))

            return enhanced

        except Exception as e:
            logger.error(f"Artistic enhancement error: {e}")
            return image

    def _save_composed_image(self, composed_img: Image.Image) -> Optional[str]:
        """
        Save the color-blended image.
        """
        try:
            filename = f"color_blended_{uuid.uuid4().hex[:8]}.jpg"
            subfolder = 'composed_scenes'
            filepath = os.path.join(settings.MEDIA_ROOT, subfolder, filename)

            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Save with high quality
            composed_img.save(filepath, 'JPEG', quality=94, optimize=True)

            return os.path.join(subfolder, filename)

        except Exception as e:
            logger.error(f"Color-blended image saving error: {e}")
            return None
