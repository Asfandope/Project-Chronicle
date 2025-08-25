"""
Layout visualization tool - creates annotated PDFs showing layout analysis results.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import fitz  # PyMuPDF
import structlog
from PIL import Image, ImageDraw, ImageFont

from .types import (
    LayoutError, LayoutResult, PageLayout, TextBlock, BlockType, 
    BoundingBox, VisualizationConfig
)


logger = structlog.get_logger(__name__)


class LayoutVisualizer:
    """
    Creates visual annotations of layout analysis results.
    
    Generates annotated PDFs and images showing detected text blocks,
    their classifications, and layout structure.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize layout visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        self.logger = logger.bind(component="LayoutVisualizer")
    
    def create_annotated_pdf(
        self, 
        original_pdf_path: Path, 
        layout_result: LayoutResult, 
        output_path: Path,
        page_range: Optional[Tuple[int, int]] = None
    ) -> Path:
        """
        Create annotated PDF with layout analysis overlays.
        
        Args:
            original_pdf_path: Path to original PDF
            layout_result: Layout analysis result
            output_path: Path for annotated PDF output
            page_range: Optional page range to annotate
            
        Returns:
            Path to created annotated PDF
            
        Raises:
            LayoutError: If visualization fails
        """
        try:
            self.logger.info("Creating annotated PDF",
                           original_pdf=str(original_pdf_path),
                           output_path=str(output_path))
            
            # Open original PDF
            try:
                doc = fitz.open(str(original_pdf_path))
            except Exception as e:
                raise LayoutError(f"Cannot open original PDF: {str(e)}")
            
            try:
                # Determine page range
                if page_range:
                    start_page, end_page = page_range
                    start_page = max(1, start_page)
                    end_page = min(doc.page_count, end_page)
                else:
                    start_page, end_page = 1, doc.page_count
                
                # Create output directory if needed
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Create new document for annotated version
                annotated_doc = fitz.open()
                
                # Process each page
                for page_num in range(start_page, end_page + 1):
                    try:
                        # Get original page
                        original_page = doc[page_num - 1]
                        
                        # Copy page to new document
                        annotated_doc.insert_pdf(doc, from_page=page_num - 1, to_page=page_num - 1)
                        annotated_page = annotated_doc[-1]
                        
                        # Get layout for this page
                        page_layout = layout_result.get_page(page_num)
                        if page_layout:
                            self._annotate_page(annotated_page, page_layout)
                        
                    except Exception as e:
                        self.logger.error("Error annotating page",
                                        page_num=page_num, error=str(e))
                        continue
                
                # Save annotated PDF
                annotated_doc.save(str(output_path))
                annotated_doc.close()
                
                self.logger.info("Annotated PDF created successfully",
                               output_path=str(output_path),
                               pages_annotated=end_page - start_page + 1)
                
                return output_path
                
            finally:
                doc.close()
                
        except LayoutError:
            raise
        except Exception as e:
            self.logger.error("Unexpected error creating annotated PDF",
                            error=str(e), exc_info=True)
            raise LayoutError(f"Annotated PDF creation failed: {str(e)}")
    
    def _annotate_page(self, page: fitz.Page, page_layout: PageLayout):
        """Annotate a single page with layout information."""
        try:
            # Add title annotation
            self._add_page_title_annotation(page, page_layout)
            
            # Annotate each text block
            for i, block in enumerate(page_layout.text_blocks):
                self._annotate_text_block(page, block, i)
            
            # Add legend
            if self.config.show_reading_order or self.config.show_confidence:
                self._add_legend(page, page_layout)
                
        except Exception as e:
            self.logger.warning("Error annotating page",
                              page_num=page_layout.page_num, error=str(e))
    
    def _add_page_title_annotation(self, page: fitz.Page, page_layout: PageLayout):
        """Add page-level annotation with summary information."""
        try:
            rect = page.rect
            
            # Add summary text at top of page
            summary_text = (
                f"Page {page_layout.page_num} - "
                f"{len(page_layout.text_blocks)} blocks - "
                f"Processing: {page_layout.processing_time:.2f}s"
            )
            
            # Position at top of page
            text_rect = fitz.Rect(10, 5, rect.width - 10, 25)
            
            # Add background rectangle
            page.draw_rect(text_rect.expand(2), color=(1, 1, 1), fill=(1, 1, 1))
            page.draw_rect(text_rect.expand(2), color=(0, 0, 0), width=1)
            
            # Add text
            page.insert_text(
                text_rect.tl + (2, 12),
                summary_text,
                fontsize=10,
                color=(0, 0, 0)
            )
            
        except Exception as e:
            self.logger.warning("Error adding page title annotation", error=str(e))
    
    def _annotate_text_block(self, page: fitz.Page, block: TextBlock, block_index: int):
        """Annotate a single text block."""
        try:
            # Convert bounding box to fitz.Rect
            rect = fitz.Rect(block.bbox.x0, block.bbox.y0, block.bbox.x1, block.bbox.y1)
            
            # Get color for block type
            color_hex = self.config.get_color(block.block_type)
            color_rgb = self._hex_to_rgb(color_hex)
            
            # Draw bounding box if enabled
            if self.config.show_bounding_boxes:
                page.draw_rect(
                    rect,
                    color=color_rgb,
                    fill=(*color_rgb, self.config.box_opacity),
                    width=self.config.line_width
                )
            
            # Add block type label
            label_text = block.block_type.value.upper()
            if self.config.show_confidence:
                label_text += f" ({block.confidence:.2f})"
            if self.config.show_reading_order:
                label_text += f" #{block.reading_order}"
            
            # Position label at top-left of block
            label_pos = rect.tl + (2, 10)
            
            # Add label background
            label_rect = fitz.Rect(label_pos.x - 1, label_pos.y - 8, 
                                 label_pos.x + len(label_text) * 5, label_pos.y + 2)
            page.draw_rect(label_rect, color=(1, 1, 1), fill=(1, 1, 1))
            
            # Add label text
            page.insert_text(
                label_pos,
                label_text,
                fontsize=self.config.text_size,
                color=(0, 0, 0)
            )
            
            # Add text content preview if enabled and space allows
            if (self.config.show_text and 
                rect.height > 30 and rect.width > 100):
                self._add_text_preview(page, block, rect)
                
        except Exception as e:
            self.logger.warning("Error annotating text block",
                              block_index=block_index, error=str(e))
    
    def _add_text_preview(self, page: fitz.Page, block: TextBlock, rect: fitz.Rect):
        """Add text content preview inside block."""
        try:
            # Get text preview (first few words)
            preview_text = block.text[:50] + "..." if len(block.text) > 50 else block.text
            preview_text = preview_text.replace('\n', ' ')
            
            # Position text inside block
            text_pos = rect.tl + (5, 25)
            
            # Calculate available space
            available_width = rect.width - 10
            available_height = rect.height - 30
            
            if available_width > 50 and available_height > 15:
                # Insert text with word wrapping
                page.insert_textbox(
                    fitz.Rect(text_pos.x, text_pos.y, 
                            text_pos.x + available_width, text_pos.y + available_height),
                    preview_text,
                    fontsize=max(6, min(self.config.text_size - 2, 8)),
                    color=(0.3, 0.3, 0.3),
                    align=fitz.TEXT_ALIGN_LEFT
                )
                
        except Exception as e:
            self.logger.warning("Error adding text preview", error=str(e))
    
    def _add_legend(self, page: fitz.Page, page_layout: PageLayout):
        """Add legend showing block type colors."""
        try:
            rect = page.rect
            
            # Get unique block types on this page
            block_types = set(block.block_type for block in page_layout.text_blocks)
            block_types = sorted(block_types, key=lambda bt: bt.value)
            
            if not block_types:
                return
            
            # Legend position (bottom right)
            legend_width = 150
            legend_height = len(block_types) * 20 + 20
            legend_rect = fitz.Rect(
                rect.width - legend_width - 10,
                rect.height - legend_height - 10,
                rect.width - 10,
                rect.height - 10
            )
            
            # Draw legend background
            page.draw_rect(legend_rect, color=(1, 1, 1), fill=(1, 1, 1))
            page.draw_rect(legend_rect, color=(0, 0, 0), width=1)
            
            # Add legend title
            page.insert_text(
                legend_rect.tl + (5, 15),
                "Block Types",
                fontsize=10,
                color=(0, 0, 0)
            )
            
            # Add legend items
            for i, block_type in enumerate(block_types):
                y_offset = 25 + i * 18
                item_pos = legend_rect.tl + (5, y_offset)
                
                # Color box
                color_hex = self.config.get_color(block_type)
                color_rgb = self._hex_to_rgb(color_hex)
                color_rect = fitz.Rect(item_pos.x, item_pos.y - 8, item_pos.x + 12, item_pos.y + 4)
                page.draw_rect(color_rect, color=color_rgb, fill=color_rgb)
                page.draw_rect(color_rect, color=(0, 0, 0), width=0.5)
                
                # Label
                page.insert_text(
                    item_pos + (15, 0),
                    block_type.value.title(),
                    fontsize=8,
                    color=(0, 0, 0)
                )
                
        except Exception as e:
            self.logger.warning("Error adding legend", error=str(e))
    
    def create_layout_images(
        self, 
        original_pdf_path: Path, 
        layout_result: LayoutResult, 
        output_dir: Path,
        page_range: Optional[Tuple[int, int]] = None
    ) -> List[Path]:
        """
        Create individual page images with layout annotations.
        
        Args:
            original_pdf_path: Path to original PDF
            layout_result: Layout analysis result
            output_dir: Directory for output images
            page_range: Optional page range to process
            
        Returns:
            List of created image file paths
        """
        try:
            self.logger.info("Creating layout images",
                           original_pdf=str(original_pdf_path),
                           output_dir=str(output_dir))
            
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Open original PDF
            doc = fitz.open(str(original_pdf_path))
            
            try:
                # Determine page range
                if page_range:
                    start_page, end_page = page_range
                    start_page = max(1, start_page)
                    end_page = min(doc.page_count, end_page)
                else:
                    start_page, end_page = 1, doc.page_count
                
                created_images = []
                
                # Process each page
                for page_num in range(start_page, end_page + 1):
                    try:
                        page = doc[page_num - 1]
                        
                        # Render page to image
                        matrix = fitz.Matrix(self.config.dpi / 72, self.config.dpi / 72)
                        pix = page.get_pixmap(matrix=matrix)
                        
                        # Convert to PIL Image
                        img_data = pix.tobytes("png")
                        pil_image = Image.open(io.BytesIO(img_data))
                        
                        # Get layout for this page
                        page_layout = layout_result.get_page(page_num)
                        if page_layout:
                            # Add annotations
                            annotated_image = self._annotate_image(pil_image, page_layout, matrix)
                        else:
                            annotated_image = pil_image
                        
                        # Save image
                        output_filename = f"page_{page_num:03d}_layout.png"
                        output_path = output_dir / output_filename
                        annotated_image.save(output_path)
                        created_images.append(output_path)
                        
                        pix = None  # Clean up
                        
                    except Exception as e:
                        self.logger.error("Error creating layout image for page",
                                        page_num=page_num, error=str(e))
                        continue
                
                self.logger.info("Layout images created",
                               images_created=len(created_images))
                
                return created_images
                
            finally:
                doc.close()
                
        except Exception as e:
            self.logger.error("Error creating layout images", error=str(e), exc_info=True)
            raise LayoutError(f"Layout image creation failed: {str(e)}")
    
    def _annotate_image(self, image: Image.Image, page_layout: PageLayout, matrix: fitz.Matrix) -> Image.Image:
        """Annotate PIL image with layout information."""
        try:
            # Create a copy for annotation
            annotated = image.copy()
            draw = ImageDraw.Draw(annotated, 'RGBA')
            
            # Scale factor for coordinates
            scale_x = matrix.a  # x scaling factor
            scale_y = matrix.d  # y scaling factor
            
            # Draw each text block
            for block in page_layout.text_blocks:
                # Scale coordinates
                x0 = block.bbox.x0 * scale_x
                y0 = block.bbox.y0 * scale_y
                x1 = block.bbox.x1 * scale_x
                y1 = block.bbox.y1 * scale_y
                
                # Get color
                color_hex = self.config.get_color(block.block_type)
                color_rgb = self._hex_to_rgb(color_hex)
                color_rgba = (*color_rgb, int(255 * self.config.box_opacity))
                
                # Draw bounding box
                if self.config.show_bounding_boxes:
                    draw.rectangle([x0, y0, x1, y1], outline=color_rgb, 
                                 fill=color_rgba, width=int(self.config.line_width))
                
                # Add label
                label = block.block_type.value.upper()
                if self.config.show_confidence:
                    label += f" ({block.confidence:.2f})"
                
                # Try to load a font, fall back to default if not available
                try:
                    font = ImageFont.truetype("arial.ttf", self.config.text_size)
                except (OSError, IOError):
                    font = ImageFont.load_default()
                
                # Draw label background
                bbox = draw.textbbox((x0 + 2, y0 + 2), label, font=font)
                draw.rectangle(bbox, fill=(255, 255, 255, 200))
                
                # Draw label text
                draw.text((x0 + 2, y0 + 2), label, fill=(0, 0, 0), font=font)
            
            return annotated
            
        except Exception as e:
            self.logger.warning("Error annotating image", error=str(e))
            return image
    
    def create_layout_summary_image(self, layout_result: LayoutResult, output_path: Path) -> Path:
        """
        Create summary image showing layout statistics.
        
        Args:
            layout_result: Layout analysis result
            output_path: Path for summary image
            
        Returns:
            Path to created summary image
        """
        try:
            # Create summary data
            summary_data = self._prepare_summary_data(layout_result)
            
            # Create image
            img_width = 800
            img_height = 600
            image = Image.new('RGB', (img_width, img_height), 'white')
            draw = ImageDraw.Draw(image)
            
            # Try to load a font
            try:
                title_font = ImageFont.truetype("arial.ttf", 24)
                text_font = ImageFont.truetype("arial.ttf", 14)
            except (OSError, IOError):
                title_font = ImageFont.load_default()
                text_font = ImageFont.load_default()
            
            # Draw title
            title = f"Layout Analysis Summary - {layout_result.pdf_path.name}"
            draw.text((20, 20), title, fill='black', font=title_font)
            
            # Draw summary statistics
            y_pos = 70
            line_height = 25
            
            stats_text = [
                f"Pages analyzed: {layout_result.page_count}",
                f"Total blocks: {layout_result.total_blocks}",
                f"Processing time: {layout_result.total_processing_time:.2f}s",
                "",
                "Blocks by type:"
            ]
            
            for block_type, count in layout_result.blocks_by_type_total.items():
                stats_text.append(f"  {block_type.title()}: {count}")
            
            for line in stats_text:
                draw.text((20, y_pos), line, fill='black', font=text_font)
                y_pos += line_height
            
            # Save image
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path)
            
            return output_path
            
        except Exception as e:
            self.logger.error("Error creating summary image", error=str(e))
            raise LayoutError(f"Summary image creation failed: {str(e)}")
    
    def _prepare_summary_data(self, layout_result: LayoutResult) -> Dict[str, Any]:
        """Prepare summary data for visualization."""
        try:
            return {
                "total_pages": layout_result.page_count,
                "total_blocks": layout_result.total_blocks,
                "processing_time": layout_result.total_processing_time,
                "blocks_by_type": layout_result.blocks_by_type_total,
                "avg_blocks_per_page": layout_result.total_blocks / max(layout_result.page_count, 1)
            }
        except Exception as e:
            self.logger.warning("Error preparing summary data", error=str(e))
            return {}
    
    def _hex_to_rgb(self, hex_color: str) -> Tuple[float, float, float]:
        """Convert hex color to RGB tuple (0-1 range for fitz, 0-255 for PIL)."""
        try:
            hex_color = hex_color.lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            # Return as 0-1 range for fitz
            return (rgb[0]/255, rgb[1]/255, rgb[2]/255)
        except Exception:
            return (0.5, 0.5, 0.5)  # Default gray
    
    def export_layout_data(self, layout_result: LayoutResult, output_path: Path, format: str = "json"):
        """
        Export layout data in various formats.
        
        Args:
            layout_result: Layout analysis result
            output_path: Output file path
            format: Export format ("json", "csv", "xml")
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == "json":
                layout_result.save_json(output_path)
            
            elif format.lower() == "csv":
                self._export_csv(layout_result, output_path)
            
            elif format.lower() == "xml":
                self._export_xml(layout_result, output_path)
            
            else:
                raise LayoutError(f"Unsupported export format: {format}")
                
        except Exception as e:
            self.logger.error("Error exporting layout data", 
                            format=format, error=str(e))
            raise LayoutError(f"Layout data export failed: {str(e)}")
    
    def _export_csv(self, layout_result: LayoutResult, output_path: Path):
        """Export layout data as CSV."""
        import csv
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'page_num', 'block_type', 'text', 'x0', 'y0', 'x1', 'y1',
                'width', 'height', 'confidence', 'font_size', 'is_bold', 
                'is_italic', 'reading_order', 'column'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for page in layout_result.pages:
                for block in page.text_blocks:
                    writer.writerow({
                        'page_num': block.page_num,
                        'block_type': block.block_type.value,
                        'text': block.text.replace('\n', ' '),
                        'x0': block.bbox.x0,
                        'y0': block.bbox.y0,
                        'x1': block.bbox.x1,
                        'y1': block.bbox.y1,
                        'width': block.bbox.width,
                        'height': block.bbox.height,
                        'confidence': block.confidence,
                        'font_size': block.font_size,
                        'is_bold': block.is_bold,
                        'is_italic': block.is_italic,
                        'reading_order': block.reading_order,
                        'column': block.column
                    })
    
    def _export_xml(self, layout_result: LayoutResult, output_path: Path):
        """Export layout data as XML."""
        import xml.etree.ElementTree as ET
        
        root = ET.Element("layout_analysis")
        root.set("pdf_path", str(layout_result.pdf_path))
        root.set("page_count", str(layout_result.page_count))
        root.set("total_blocks", str(layout_result.total_blocks))
        
        for page in layout_result.pages:
            page_elem = ET.SubElement(root, "page")
            page_elem.set("number", str(page.page_num))
            page_elem.set("width", str(page.page_width))
            page_elem.set("height", str(page.page_height))
            
            for block in page.text_blocks:
                block_elem = ET.SubElement(page_elem, "text_block")
                block_elem.set("type", block.block_type.value)
                block_elem.set("confidence", str(block.confidence))
                block_elem.set("reading_order", str(block.reading_order))
                
                # Bounding box
                bbox_elem = ET.SubElement(block_elem, "bounding_box")
                bbox_elem.set("x0", str(block.bbox.x0))
                bbox_elem.set("y0", str(block.bbox.y0))
                bbox_elem.set("x1", str(block.bbox.x1))
                bbox_elem.set("y1", str(block.bbox.y1))
                
                # Text content
                text_elem = ET.SubElement(block_elem, "text")
                text_elem.text = block.text
        
        # Write XML
        tree = ET.ElementTree(root)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)