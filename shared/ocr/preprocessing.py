"""
Image preprocessing pipeline for scanned PDFs.
Implements brand-specific preprocessing from YAML configurations.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import structlog
import yaml
from PIL import Image

from .types import OCRError, PreprocessingConfig

logger = structlog.get_logger(__name__)


class ImagePreprocessor:
    """
    Advanced image preprocessing pipeline for OCR optimization.

    Implements brand-specific preprocessing strategies to achieve <2% WER
    on scanned documents.
    """

    def __init__(self, brand_configs_dir: Optional[Path] = None):
        """
        Initialize image preprocessor.

        Args:
            brand_configs_dir: Directory containing brand-specific YAML configs
        """
        self.brand_configs_dir = brand_configs_dir
        self.logger = logger.bind(component="ImagePreprocessor")
        self.brand_configs = {}

        # Load brand-specific configurations
        if brand_configs_dir:
            self._load_brand_configs()

    def _load_brand_configs(self):
        """Load brand-specific preprocessing configurations from YAML files."""
        try:
            if not self.brand_configs_dir or not self.brand_configs_dir.exists():
                self.logger.warning(
                    "Brand configs directory not found", dir=str(self.brand_configs_dir)
                )
                return

            for config_file in self.brand_configs_dir.glob("*.yaml"):
                try:
                    with open(config_file, "r") as f:
                        config_data = yaml.safe_load(f)

                    brand_name = config_file.stem

                    # Extract preprocessing configuration if it exists
                    if "preprocessing" in config_data:
                        preprocessing_config = PreprocessingConfig(
                            **config_data["preprocessing"]
                        )
                        self.brand_configs[brand_name] = preprocessing_config

                        self.logger.info(
                            "Loaded brand preprocessing config",
                            brand=brand_name,
                            config_file=str(config_file),
                        )

                except Exception as e:
                    self.logger.error(
                        "Error loading brand config",
                        config_file=str(config_file),
                        error=str(e),
                    )

            self.logger.info(
                "Brand configs loaded", brands=list(self.brand_configs.keys())
            )

        except Exception as e:
            self.logger.error("Error loading brand configurations", error=str(e))

    def get_brand_config(self, brand: str) -> PreprocessingConfig:
        """Get preprocessing configuration for a specific brand."""
        return self.brand_configs.get(brand, PreprocessingConfig())

    def process_image(
        self,
        image: Image.Image,
        config: Optional[PreprocessingConfig] = None,
        brand: Optional[str] = None,
    ) -> Tuple[Image.Image, List[str]]:
        """
        Process image with brand-specific preprocessing pipeline.

        Args:
            image: Input PIL Image
            config: Preprocessing configuration
            brand: Brand name for configuration override

        Returns:
            Tuple of (processed_image, applied_steps)

        Raises:
            OCRError: If preprocessing fails
        """
        start_time = time.time()
        applied_steps = []

        try:
            # Get configuration
            if brand and brand in self.brand_configs:
                config = self.brand_configs[brand]
            elif config is None:
                config = PreprocessingConfig()

            self.logger.debug(
                "Starting image preprocessing", brand=brand, image_size=image.size
            )

            # Convert to numpy array for processing
            img_array = np.array(image)
            original_image = img_array.copy()

            # Step 1: Border removal
            if config.border_removal:
                img_array = self._remove_borders(img_array, config)
                applied_steps.append("border_removal")

            # Step 2: Noise reduction
            if config.denoise_enabled:
                img_array = self._denoise_image(img_array, config)
                applied_steps.append("denoise")

            # Step 3: Deskewing
            if config.deskew_enabled:
                img_array, angle = self._deskew_image(img_array, config)
                if abs(angle) > config.deskew_angle_threshold:
                    applied_steps.append(f"deskew_{angle:.1f}deg")

            # Step 4: DPI normalization and scaling
            img_array = self._normalize_resolution(img_array, config)
            applied_steps.append("resolution_norm")

            # Step 5: Contrast enhancement
            if config.contrast_enhancement:
                img_array = self._enhance_contrast(img_array, config)
                applied_steps.append("contrast_enhance")

            # Step 6: Adaptive thresholding
            if config.adaptive_threshold:
                img_array = self._adaptive_threshold(img_array, config)
                applied_steps.append("adaptive_threshold")

            # Step 7: Morphological operations
            if config.morphology_enabled:
                img_array = self._morphological_cleanup(img_array, config)
                applied_steps.append("morphology")

            # Step 8: Quality-based selection
            if config.auto_select_best:
                img_array = self._select_best_version(original_image, img_array, config)
                applied_steps.append("quality_selection")

            # Convert back to PIL Image
            processed_image = Image.fromarray(img_array)

            processing_time = time.time() - start_time

            self.logger.debug(
                "Image preprocessing completed",
                brand=brand,
                applied_steps=applied_steps,
                processing_time=processing_time,
            )

            return processed_image, applied_steps

        except Exception as e:
            self.logger.error(
                "Error in image preprocessing", brand=brand, error=str(e), exc_info=True
            )
            raise OCRError(f"Image preprocessing failed: {str(e)}")

    def _remove_borders(
        self, img_array: np.ndarray, config: PreprocessingConfig
    ) -> np.ndarray:
        """Remove borders and margins from scanned images."""
        try:
            h, w = img_array.shape[:2]

            # Convert to grayscale if needed
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array.copy()

            # Find content boundaries using edge detection
            edges = cv2.Canny(gray, 50, 150)

            # Find contours
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                # Find the largest contour (main content)
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w_content, h_content = cv2.boundingRect(largest_contour)

                # Add some padding
                padding = int(min(h, w) * config.border_threshold)
                x = max(0, x - padding)
                y = max(0, y - padding)
                w_content = min(w - x, w_content + 2 * padding)
                h_content = min(h - y, h_content + 2 * padding)

                # Crop to content area
                if len(img_array.shape) == 3:
                    return img_array[y : y + h_content, x : x + w_content]
                else:
                    return img_array[y : y + h_content, x : x + w_content]

            return img_array

        except Exception as e:
            self.logger.warning("Error in border removal", error=str(e))
            return img_array

    def _denoise_image(
        self, img_array: np.ndarray, config: PreprocessingConfig
    ) -> np.ndarray:
        """Apply noise reduction techniques."""
        try:
            # Convert to grayscale if needed
            if len(img_array.shape) == 3:
                # Apply denoising to each channel
                denoised = cv2.fastNlMeansDenoisingColored(
                    img_array,
                    None,
                    config.denoise_strength,
                    config.denoise_strength,
                    7,
                    21,
                )
            else:
                # Grayscale denoising
                denoised = cv2.fastNlMeansDenoising(
                    img_array, None, config.denoise_strength, 7, 21
                )

            # Additional Gaussian blur for smoothing
            if config.gaussian_blur_kernel > 0:
                kernel_size = config.gaussian_blur_kernel * 2 + 1  # Ensure odd number
                denoised = cv2.GaussianBlur(denoised, (kernel_size, kernel_size), 0)

            return denoised

        except Exception as e:
            self.logger.warning("Error in denoising", error=str(e))
            return img_array

    def _deskew_image(
        self, img_array: np.ndarray, config: PreprocessingConfig
    ) -> Tuple[np.ndarray, float]:
        """Detect and correct skew in scanned images."""
        try:
            # Convert to grayscale if needed
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array.copy()

            # Edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)

            # Hough line detection
            lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

            if lines is not None:
                angles = []
                for rho, theta in lines[:, 0]:
                    angle = theta * 180 / np.pi
                    # Convert to skew angle (-90 to 90 degrees)
                    if angle > 90:
                        angle -= 180
                    angles.append(angle)

                # Find most common angle (mode)
                if angles:
                    angle_hist, bins = np.histogram(angles, bins=180, range=(-90, 90))
                    most_common_angle = bins[np.argmax(angle_hist)]

                    # Only deskew if angle is significant
                    if (
                        abs(most_common_angle) > config.deskew_angle_threshold
                        and abs(most_common_angle) < config.deskew_max_angle
                    ):
                        # Rotate image
                        if len(img_array.shape) == 3:
                            h, w, c = img_array.shape
                            center = (w // 2, h // 2)
                            rotation_matrix = cv2.getRotationMatrix2D(
                                center, most_common_angle, 1.0
                            )
                            deskewed = cv2.warpAffine(
                                img_array,
                                rotation_matrix,
                                (w, h),
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE,
                            )
                        else:
                            h, w = img_array.shape
                            center = (w // 2, h // 2)
                            rotation_matrix = cv2.getRotationMatrix2D(
                                center, most_common_angle, 1.0
                            )
                            deskewed = cv2.warpAffine(
                                img_array,
                                rotation_matrix,
                                (w, h),
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE,
                            )

                        return deskewed, most_common_angle

            return img_array, 0.0

        except Exception as e:
            self.logger.warning("Error in deskewing", error=str(e))
            return img_array, 0.0

    def _normalize_resolution(
        self, img_array: np.ndarray, config: PreprocessingConfig
    ) -> np.ndarray:
        """Normalize image resolution for optimal OCR."""
        try:
            h, w = img_array.shape[:2]

            # Estimate current DPI (simplified)
            estimated_dpi = min(w, h) / 8.5  # Assume letter size width

            if estimated_dpi < config.min_dpi:
                # Upscale image
                scale_factor = config.target_dpi / estimated_dpi
                scale_factor = min(scale_factor, config.upscale_factor)  # Limit scaling

                new_w = int(w * scale_factor)
                new_h = int(h * scale_factor)

                if len(img_array.shape) == 3:
                    upscaled = cv2.resize(
                        img_array, (new_w, new_h), interpolation=cv2.INTER_CUBIC
                    )
                else:
                    upscaled = cv2.resize(
                        img_array, (new_w, new_h), interpolation=cv2.INTER_CUBIC
                    )

                return upscaled

            return img_array

        except Exception as e:
            self.logger.warning("Error in resolution normalization", error=str(e))
            return img_array

    def _enhance_contrast(
        self, img_array: np.ndarray, config: PreprocessingConfig
    ) -> np.ndarray:
        """Enhance image contrast for better text recognition."""
        try:
            if len(img_array.shape) == 3:
                # Convert to LAB color space for better contrast enhancement
                lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)

                # Apply CLAHE to L channel
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l = clahe.apply(l)

                # Merge channels and convert back to RGB
                enhanced_lab = cv2.merge([l, a, b])
                enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
            else:
                # Grayscale CLAHE
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(img_array)

            return enhanced

        except Exception as e:
            self.logger.warning("Error in contrast enhancement", error=str(e))
            return img_array

    def _adaptive_threshold(
        self, img_array: np.ndarray, config: PreprocessingConfig
    ) -> np.ndarray:
        """Apply adaptive thresholding for better text separation."""
        try:
            # Convert to grayscale if needed
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array.copy()

            # Apply adaptive threshold
            binary = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                config.threshold_block_size,
                config.threshold_constant,
            )

            # If input was color, convert back to 3-channel
            if len(img_array.shape) == 3:
                binary_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
                return binary_color
            else:
                return binary

        except Exception as e:
            self.logger.warning("Error in adaptive thresholding", error=str(e))
            return img_array

    def _morphological_cleanup(
        self, img_array: np.ndarray, config: PreprocessingConfig
    ) -> np.ndarray:
        """Apply morphological operations to clean up text."""
        try:
            # Convert to grayscale if needed
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                is_color = True
            else:
                gray = img_array.copy()
                is_color = False

            # Define morphological kernel
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (config.kernel_size, config.kernel_size)
            )

            # Closing operation (fill gaps in characters)
            if config.closing_iterations > 0:
                gray = cv2.morphologyEx(
                    gray, cv2.MORPH_CLOSE, kernel, iterations=config.closing_iterations
                )

            # Opening operation (remove noise)
            if config.opening_iterations > 0:
                gray = cv2.morphologyEx(
                    gray, cv2.MORPH_OPEN, kernel, iterations=config.opening_iterations
                )

            # Convert back to color if needed
            if is_color:
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            else:
                return gray

        except Exception as e:
            self.logger.warning("Error in morphological cleanup", error=str(e))
            return img_array

    def _select_best_version(
        self, original: np.ndarray, processed: np.ndarray, config: PreprocessingConfig
    ) -> np.ndarray:
        """Select the best version based on quality metrics."""
        try:
            # Calculate quality metrics for both versions
            original_metrics = self._calculate_image_quality(original, config)
            processed_metrics = self._calculate_image_quality(processed, config)

            # Weighted scoring
            original_score = self._calculate_quality_score(original_metrics, config)
            processed_score = self._calculate_quality_score(processed_metrics, config)

            # Select best version
            if processed_score > original_score:
                self.logger.debug(
                    "Selected processed version",
                    original_score=original_score,
                    processed_score=processed_score,
                )
                return processed
            else:
                self.logger.debug(
                    "Selected original version",
                    original_score=original_score,
                    processed_score=processed_score,
                )
                return original

        except Exception as e:
            self.logger.warning("Error in quality selection", error=str(e))
            return processed  # Default to processed version

    def _calculate_image_quality(
        self, img_array: np.ndarray, config: PreprocessingConfig
    ) -> Dict[str, float]:
        """Calculate image quality metrics."""
        try:
            # Convert to grayscale if needed
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array.copy()

            metrics = {}

            # Sharpness (variance of Laplacian)
            if "sharpness" in config.quality_metrics:
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                metrics["sharpness"] = laplacian_var

            # Contrast (standard deviation)
            if "contrast" in config.quality_metrics:
                contrast = gray.std()
                metrics["contrast"] = contrast

            # Noise level (estimate using high-frequency content)
            if "noise_level" in config.quality_metrics:
                # Apply high-pass filter
                kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
                high_freq = cv2.filter2D(gray, -1, kernel)
                noise_level = np.mean(np.abs(high_freq))
                metrics["noise_level"] = noise_level

            return metrics

        except Exception as e:
            self.logger.warning("Error calculating image quality", error=str(e))
            return {"sharpness": 0, "contrast": 0, "noise_level": 0}

    def _calculate_quality_score(
        self, metrics: Dict[str, float], config: PreprocessingConfig
    ) -> float:
        """Calculate overall quality score from metrics."""
        try:
            score = 0.0

            # Sharpness (higher is better)
            if "sharpness" in metrics:
                sharpness_score = min(1.0, metrics["sharpness"] / 1000.0)  # Normalize
                score += sharpness_score * 0.4

            # Contrast (higher is better, up to a point)
            if "contrast" in metrics:
                contrast_score = min(1.0, metrics["contrast"] / 100.0)  # Normalize
                score += contrast_score * 0.4

            # Noise level (lower is better)
            if "noise_level" in metrics:
                noise_score = max(
                    0.0, 1.0 - (metrics["noise_level"] / 50.0)
                )  # Invert and normalize
                score += noise_score * 0.2

            return score

        except Exception as e:
            self.logger.warning("Error calculating quality score", error=str(e))
            return 0.5

    def process_multiple_images(
        self,
        images: List[Image.Image],
        config: Optional[PreprocessingConfig] = None,
        brand: Optional[str] = None,
    ) -> List[Tuple[Image.Image, List[str]]]:
        """Process multiple images with the same configuration."""
        results = []

        for i, image in enumerate(images):
            try:
                processed_image, applied_steps = self.process_image(
                    image, config, brand
                )
                results.append((processed_image, applied_steps))
            except Exception as e:
                self.logger.error("Error processing image", image_index=i, error=str(e))
                results.append((image, ["error"]))  # Return original on error

        return results

    def get_preprocessing_summary(self, applied_steps: List[str]) -> Dict[str, Any]:
        """Get summary of applied preprocessing steps."""
        return {
            "total_steps": len(applied_steps),
            "applied_steps": applied_steps,
            "has_denoising": any("denoise" in step for step in applied_steps),
            "has_deskewing": any("deskew" in step for step in applied_steps),
            "has_contrast_enhancement": any(
                "contrast" in step for step in applied_steps
            ),
            "has_thresholding": any("threshold" in step for step in applied_steps),
            "has_morphology": any("morphology" in step for step in applied_steps),
        }
