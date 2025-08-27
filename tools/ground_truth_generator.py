#!/usr/bin/env python3
"""
Ground Truth Generator for Magazine Extraction Gold Standard Data

This tool generates XML ground truth files following the magazine extraction schema v1.0.
It provides both automated and manual annotation workflows.
"""

import argparse
import json
import logging
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from xml.dom import minidom

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Bounding box coordinates"""

    x: float
    y: float
    width: float
    height: float


@dataclass
class Block:
    """Text or content block"""

    id: str
    type: str  # title, body, caption, byline, ad, etc.
    text: str
    bbox: BoundingBox
    page: int
    confidence: float
    font_info: Optional[Dict[str, Any]] = None


@dataclass
class Article:
    """Complete article structure"""

    id: str
    title: str
    title_blocks: List[str]  # Block IDs
    body_blocks: List[str]  # Block IDs
    byline_blocks: List[str]  # Block IDs
    caption_blocks: List[str]  # Block IDs
    contributors: List[Dict[str, Any]]
    images: List[Dict[str, Any]]
    pages: List[int]
    confidence: float


@dataclass
class Page:
    """Page structure with blocks"""

    number: int
    width: float
    height: float
    blocks: List[Block]


class GroundTruthGenerator:
    """Main class for generating ground truth XML files"""

    def __init__(self, schema_version: str = "v1.0"):
        self.schema_version = schema_version

        # Common block type classifications
        self.block_types = [
            "title",
            "subtitle",
            "body",
            "byline",
            "caption",
            "ad",
            "header",
            "footer",
            "sidebar",
            "pullquote",
            "image",
            "table",
            "chart",
            "divider",
        ]

        # Common contributor roles
        self.contributor_roles = [
            "author",
            "photographer",
            "illustrator",
            "editor",
            "correspondent",
            "columnist",
            "reviewer",
        ]

    def create_ground_truth_template(
        self, brand: str, issue_id: str, pages_count: int
    ) -> ET.Element:
        """Create empty ground truth XML template"""

        root = ET.Element("magazine_extraction")
        root.set("schema_version", self.schema_version)
        root.set("brand", brand)
        root.set("issue_id", issue_id)
        root.set("generation_time", datetime.now().isoformat())

        # Metadata section
        metadata = ET.SubElement(root, "metadata")
        ET.SubElement(metadata, "brand").text = brand
        ET.SubElement(metadata, "issue_id").text = issue_id
        ET.SubElement(metadata, "schema_version").text = self.schema_version
        ET.SubElement(metadata, "total_pages").text = str(pages_count)
        ET.SubElement(metadata, "extraction_date").text = datetime.now().isoformat()
        ET.SubElement(metadata, "annotator").text = "manual"
        ET.SubElement(metadata, "validation_status").text = "pending"

        # Pages section
        pages_elem = ET.SubElement(root, "pages")
        for i in range(1, pages_count + 1):
            page_elem = ET.SubElement(pages_elem, "page")
            page_elem.set("number", str(i))
            page_elem.set("width", "612")  # Standard letter size
            page_elem.set("height", "792")

            blocks_elem = ET.SubElement(page_elem, "blocks")
            # Add placeholder blocks
            self._add_placeholder_block(
                blocks_elem,
                f"block_{i}_1",
                "title",
                f"Sample Title Page {i}",
                100,
                100,
                400,
                50,
                0.95,
            )
            self._add_placeholder_block(
                blocks_elem,
                f"block_{i}_2",
                "body",
                f"Sample body text for page {i}...",
                100,
                200,
                400,
                200,
                0.90,
            )

        # Articles section
        articles_elem = ET.SubElement(root, "articles")
        sample_article = ET.SubElement(articles_elem, "article")
        sample_article.set("id", "article_1")
        ET.SubElement(sample_article, "title").text = "Sample Article Title"

        title_blocks = ET.SubElement(sample_article, "title_blocks")
        ET.SubElement(title_blocks, "block_id").text = "block_1_1"

        body_blocks = ET.SubElement(sample_article, "body_blocks")
        ET.SubElement(body_blocks, "block_id").text = "block_1_2"

        ET.SubElement(sample_article, "byline_blocks")
        ET.SubElement(sample_article, "caption_blocks")

        contributors = ET.SubElement(sample_article, "contributors")
        sample_contributor = ET.SubElement(contributors, "contributor")
        sample_contributor.set("id", "contributor_1")
        ET.SubElement(sample_contributor, "name").text = "Sample Author"
        ET.SubElement(sample_contributor, "role").text = "author"
        ET.SubElement(sample_contributor, "confidence").text = "0.95"

        ET.SubElement(sample_article, "images")
        pages_elem = ET.SubElement(sample_article, "pages")
        ET.SubElement(pages_elem, "page_number").text = "1"

        sample_article.set("confidence", "0.90")

        return root

    def _add_placeholder_block(
        self,
        blocks_elem: ET.Element,
        block_id: str,
        block_type: str,
        text: str,
        x: float,
        y: float,
        width: float,
        height: float,
        confidence: float,
    ):
        """Add a placeholder block to XML"""
        block_elem = ET.SubElement(blocks_elem, "block")
        block_elem.set("id", block_id)
        block_elem.set("type", block_type)
        block_elem.set("confidence", str(confidence))

        text_elem = ET.SubElement(block_elem, "text")
        text_elem.text = text

        bbox_elem = ET.SubElement(block_elem, "bbox")
        bbox_elem.set("x", str(x))
        bbox_elem.set("y", str(y))
        bbox_elem.set("width", str(width))
        bbox_elem.set("height", str(height))

        font_elem = ET.SubElement(block_elem, "font_info")
        font_elem.set("family", "Arial")
        font_elem.set("size", "12")
        font_elem.set("style", "normal")

    def load_extraction_results(self, results_path: str) -> Dict[str, Any]:
        """Load existing extraction results for ground truth generation"""
        if not os.path.exists(results_path):
            logger.warning(f"Results file not found: {results_path}")
            return {}

        try:
            with open(results_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading extraction results: {str(e)}")
            return {}

    def generate_from_extraction_results(
        self, brand: str, issue_id: str, extraction_results: Dict[str, Any]
    ) -> ET.Element:
        """Generate ground truth XML from extraction pipeline results"""

        pages_data = extraction_results.get("pages", [])
        articles_data = extraction_results.get("articles", {})
        blocks_data = extraction_results.get("blocks", {})

        pages_count = len(pages_data)
        root = ET.Element("magazine_extraction")
        root.set("schema_version", self.schema_version)
        root.set("brand", brand)
        root.set("issue_id", issue_id)
        root.set("generation_time", datetime.now().isoformat())

        # Metadata
        metadata = ET.SubElement(root, "metadata")
        ET.SubElement(metadata, "brand").text = brand
        ET.SubElement(metadata, "issue_id").text = issue_id
        ET.SubElement(metadata, "schema_version").text = self.schema_version
        ET.SubElement(metadata, "total_pages").text = str(pages_count)
        ET.SubElement(metadata, "extraction_date").text = datetime.now().isoformat()
        ET.SubElement(metadata, "annotator").text = "automated"
        ET.SubElement(metadata, "validation_status").text = "generated"

        # Pages
        pages_elem = ET.SubElement(root, "pages")
        for page_data in pages_data:
            page_num = page_data.get("page_number", 1)
            page_elem = ET.SubElement(pages_elem, "page")
            page_elem.set("number", str(page_num))
            page_elem.set("width", str(page_data.get("width", 612)))
            page_elem.set("height", str(page_data.get("height", 792)))

            blocks_elem = ET.SubElement(page_elem, "blocks")

            # Add blocks for this page
            page_blocks = page_data.get("blocks", [])
            for block_id in page_blocks:
                if block_id in blocks_data:
                    block_data = blocks_data[block_id]
                    self._add_block_from_data(blocks_elem, block_data)

        # Articles
        articles_elem = ET.SubElement(root, "articles")
        for article_id, article_data in articles_data.items():
            article_elem = ET.SubElement(articles_elem, "article")
            article_elem.set("id", article_id)
            article_elem.set("confidence", str(article_data.get("confidence", 0.8)))

            # Article title
            ET.SubElement(article_elem, "title").text = article_data.get("title", "")

            # Block references
            self._add_block_references(
                article_elem, "title_blocks", article_data.get("title_blocks", [])
            )
            self._add_block_references(
                article_elem, "body_blocks", article_data.get("body_blocks", [])
            )
            self._add_block_references(
                article_elem, "byline_blocks", article_data.get("byline_blocks", [])
            )
            self._add_block_references(
                article_elem, "caption_blocks", article_data.get("caption_blocks", [])
            )

            # Contributors
            contributors_elem = ET.SubElement(article_elem, "contributors")
            for i, contributor in enumerate(article_data.get("contributors", [])):
                contrib_elem = ET.SubElement(contributors_elem, "contributor")
                contrib_elem.set("id", f"contributor_{article_id}_{i}")
                ET.SubElement(contrib_elem, "name").text = contributor.get("name", "")
                ET.SubElement(contrib_elem, "role").text = contributor.get(
                    "role", "author"
                )
                ET.SubElement(contrib_elem, "confidence").text = str(
                    contributor.get("confidence", 0.8)
                )

            # Images
            images_elem = ET.SubElement(article_elem, "images")
            for i, image in enumerate(article_data.get("images", [])):
                img_elem = ET.SubElement(images_elem, "image")
                img_elem.set("id", f"image_{article_id}_{i}")
                ET.SubElement(img_elem, "caption").text = image.get("caption", "")
                ET.SubElement(img_elem, "source").text = image.get("source", "")

            # Pages this article appears on
            pages_elem = ET.SubElement(article_elem, "pages")
            for page_num in article_data.get("pages", []):
                ET.SubElement(pages_elem, "page_number").text = str(page_num)

        return root

    def _add_block_from_data(self, blocks_elem: ET.Element, block_data: Dict[str, Any]):
        """Add a block element from extraction data"""
        block_elem = ET.SubElement(blocks_elem, "block")
        block_elem.set("id", block_data.get("id", ""))
        block_elem.set("type", block_data.get("type", "body"))
        block_elem.set("confidence", str(block_data.get("confidence", 0.8)))

        # Text content
        text_elem = ET.SubElement(block_elem, "text")
        text_elem.text = block_data.get("text", "")

        # Bounding box
        bbox = block_data.get("bbox", [0, 0, 100, 100])
        bbox_elem = ET.SubElement(block_elem, "bbox")
        bbox_elem.set("x", str(bbox[0]))
        bbox_elem.set("y", str(bbox[1]))
        bbox_elem.set("width", str(bbox[2] - bbox[0] if len(bbox) > 2 else 100))
        bbox_elem.set("height", str(bbox[3] - bbox[1] if len(bbox) > 3 else 20))

        # Font info if available
        font_info = block_data.get("font_info", {})
        font_elem = ET.SubElement(block_elem, "font_info")
        font_elem.set("family", font_info.get("family", "Arial"))
        font_elem.set("size", str(font_info.get("size", 12)))
        font_elem.set("style", font_info.get("style", "normal"))

    def _add_block_references(
        self, article_elem: ET.Element, section_name: str, block_ids: List[str]
    ):
        """Add block ID references to article section"""
        section_elem = ET.SubElement(article_elem, section_name)
        for block_id in block_ids:
            ET.SubElement(section_elem, "block_id").text = block_id

    def validate_ground_truth(self, xml_root: ET.Element) -> Tuple[bool, List[str]]:
        """Validate generated ground truth XML"""
        issues = []

        # Check root attributes
        if not xml_root.get("schema_version"):
            issues.append("Missing schema_version attribute")
        if not xml_root.get("brand"):
            issues.append("Missing brand attribute")

        # Check required sections
        required_sections = ["metadata", "pages", "articles"]
        for section in required_sections:
            if xml_root.find(section) is None:
                issues.append(f"Missing required section: {section}")

        # Validate metadata
        metadata = xml_root.find("metadata")
        if metadata is not None:
            required_metadata = ["brand", "issue_id", "schema_version", "total_pages"]
            for field in required_metadata:
                if metadata.find(field) is None:
                    issues.append(f"Missing metadata field: {field}")

        # Validate pages structure
        pages = xml_root.find("pages")
        if pages is not None:
            page_numbers = set()
            for page in pages.findall("page"):
                page_num = page.get("number")
                if not page_num:
                    issues.append("Page missing number attribute")
                elif page_num in page_numbers:
                    issues.append(f"Duplicate page number: {page_num}")
                else:
                    page_numbers.add(page_num)

                # Check blocks
                blocks = page.find("blocks")
                if blocks is None:
                    issues.append(f"Page {page_num} missing blocks section")
                else:
                    block_ids = set()
                    for block in blocks.findall("block"):
                        block_id = block.get("id")
                        if not block_id:
                            issues.append("Block missing id attribute")
                        elif block_id in block_ids:
                            issues.append(f"Duplicate block id: {block_id}")
                        else:
                            block_ids.add(block_id)

                        # Check required block elements
                        if block.find("text") is None:
                            issues.append(f"Block {block_id} missing text element")
                        if block.find("bbox") is None:
                            issues.append(f"Block {block_id} missing bbox element")

        # Validate articles structure
        articles = xml_root.find("articles")
        if articles is not None:
            article_ids = set()
            for article in articles.findall("article"):
                article_id = article.get("id")
                if not article_id:
                    issues.append("Article missing id attribute")
                elif article_id in article_ids:
                    issues.append(f"Duplicate article id: {article_id}")
                else:
                    article_ids.add(article_id)

                # Check required article elements
                if article.find("title") is None:
                    issues.append(f"Article {article_id} missing title")

                # Check block references exist
                for section in ["title_blocks", "body_blocks"]:
                    section_elem = article.find(section)
                    if section_elem is not None:
                        for block_ref in section_elem.findall("block_id"):
                            # TODO: Validate block_id references exist in pages
                            pass

        return len(issues) == 0, issues

    def save_ground_truth(
        self, xml_root: ET.Element, output_path: str, pretty_print: bool = True
    ):
        """Save ground truth XML to file"""

        if pretty_print:
            # Pretty print the XML
            rough_string = ET.tostring(xml_root, "unicode")
            reparsed = minidom.parseString(rough_string)
            pretty_xml = reparsed.toprettyxml(indent="  ")

            # Remove extra blank lines
            pretty_lines = [line for line in pretty_xml.split("\n") if line.strip()]
            final_xml = "\n".join(pretty_lines)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_xml)
        else:
            tree = ET.ElementTree(xml_root)
            tree.write(output_path, encoding="utf-8", xml_declaration=True)

        logger.info(f"Ground truth saved to {output_path}")

    def create_annotation_template(
        self, brand: str, pdf_path: str, output_dir: str
    ) -> str:
        """Create annotation template for manual ground truth creation"""

        pdf_name = Path(pdf_path).stem
        issue_id = f"{brand}_{pdf_name}"

        # For now, assume single page - would need PDF analysis to get actual count
        pages_count = 1

        xml_root = self.create_ground_truth_template(brand, issue_id, pages_count)

        output_path = os.path.join(output_dir, f"{pdf_name}.xml")
        self.save_ground_truth(xml_root, output_path)

        # Also create annotation guide
        guide_path = os.path.join(output_dir, f"{pdf_name}_annotation_guide.txt")
        with open(guide_path, "w") as f:
            f.write(f"Annotation Guide for {pdf_name}\n")
            f.write("=" * 50 + "\n\n")
            f.write("1. Update metadata section with correct information\n")
            f.write("2. For each page, identify and annotate all blocks:\n")
            f.write("   - titles, subtitles, body text, bylines, captions\n")
            f.write("   - advertisements, headers, footers, sidebars\n")
            f.write("3. Create articles by grouping related blocks\n")
            f.write("4. Add contributor information\n")
            f.write("5. Validate XML structure when complete\n\n")
            f.write(f"Block types available: {', '.join(self.block_types)}\n")
            f.write(f"Contributor roles: {', '.join(self.contributor_roles)}\n")

        return output_path


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description="Ground Truth Generator for Magazine Extraction"
    )
    parser.add_argument(
        "command",
        choices=["template", "generate", "validate", "annotate"],
        help="Command to execute",
    )
    parser.add_argument("--brand", required=True, help="Magazine brand")
    parser.add_argument("--issue-id", help="Issue identifier")
    parser.add_argument("--pages", type=int, default=1, help="Number of pages")
    parser.add_argument("--pdf", help="PDF file path")
    parser.add_argument("--extraction-results", help="Path to extraction results JSON")
    parser.add_argument("--xml", help="Path to XML file for validation")
    parser.add_argument("--output", required=True, help="Output path")
    parser.add_argument(
        "--output-dir", help="Output directory for annotation templates"
    )

    args = parser.parse_args()

    generator = GroundTruthGenerator()

    if args.command == "template":
        issue_id = (
            args.issue_id
            or f"{args.brand}_template_{datetime.now().strftime('%Y%m%d')}"
        )
        xml_root = generator.create_ground_truth_template(
            args.brand, issue_id, args.pages
        )
        generator.save_ground_truth(xml_root, args.output)
        print(f"Template created: {args.output}")

    elif args.command == "generate" and args.extraction_results:
        issue_id = args.issue_id or f"{args.brand}_{datetime.now().strftime('%Y%m%d')}"
        extraction_results = generator.load_extraction_results(args.extraction_results)
        xml_root = generator.generate_from_extraction_results(
            args.brand, issue_id, extraction_results
        )

        # Validate before saving
        is_valid, issues = generator.validate_ground_truth(xml_root)
        if issues:
            print(f"Validation warnings: {issues}")

        generator.save_ground_truth(xml_root, args.output)
        print(f"Ground truth generated: {args.output}")

    elif args.command == "validate" and args.xml:
        tree = ET.parse(args.xml)
        xml_root = tree.getroot()
        is_valid, issues = generator.validate_ground_truth(xml_root)

        result = {"valid": is_valid, "issues": issues}
        print(json.dumps(result, indent=2))

    elif args.command == "annotate" and args.pdf and args.output_dir:
        template_path = generator.create_annotation_template(
            args.brand, args.pdf, args.output_dir
        )
        print(f"Annotation template created: {template_path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
