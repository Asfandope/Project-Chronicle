"""
Content factory for generating realistic article content and media.
"""

import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import structlog

from .types import BrandConfiguration, BrandStyle, EdgeCaseType, SyntheticDataError


logger = structlog.get_logger(__name__)


@dataclass
class ArticleTemplate:
    """Template for generating articles."""
    
    template_name: str
    article_type: str  # feature, news, review, editorial, etc.
    
    # Content structure
    typical_word_count: Tuple[int, int]  # min, max
    paragraph_count: Tuple[int, int]
    typical_images: Tuple[int, int]
    
    # Title patterns
    title_patterns: List[str]
    
    # Content themes
    content_themes: List[str]
    
    # Contributor patterns
    contributor_patterns: List[Dict[str, str]]


@dataclass
class MediaTemplate:
    """Template for generating media elements."""
    
    media_type: str  # photo, illustration, diagram, chart
    typical_dimensions: Tuple[int, int]
    caption_patterns: List[str]
    credit_patterns: List[str]


class ContentFactory:
    """
    Generates realistic article content with proper structure and metadata.
    
    Creates authentic magazine content with appropriate titles, bylines,
    body text, captions, and contributor information.
    """
    
    def __init__(self):
        self.logger = logger.bind(component="ContentFactory")
        
        # Initialize content templates
        self.article_templates = self._create_article_templates()
        self.media_templates = self._create_media_templates()
        
        # Content databases
        self.sample_content = self._load_sample_content()
        self.contributor_names = self._load_contributor_names()
        self.company_names = self._load_company_names()
        
        # Generation state
        self._used_titles = set()
        self._contributor_counter = {}
        
        self.logger.info("Content factory initialized",
                        article_templates=len(self.article_templates),
                        media_templates=len(self.media_templates))
    
    def _create_article_templates(self) -> Dict[str, ArticleTemplate]:
        """Create article templates for different types of content."""
        
        templates = {}
        
        # Tech article template
        templates["tech_feature"] = ArticleTemplate(
            template_name="tech_feature",
            article_type="feature",
            typical_word_count=(800, 2000),
            paragraph_count=(8, 15),
            typical_images=(2, 4),
            title_patterns=[
                "The Future of {technology}",
                "How {company} is Revolutionizing {field}",
                "{technology}: A Game Changer for {industry}",
                "Breaking: {technology} Reaches New Milestone",
                "Why {technology} Matters for {audience}"
            ],
            content_themes=[
                "artificial intelligence", "machine learning", "blockchain", "cloud computing",
                "cybersecurity", "data analytics", "software development", "innovation"
            ],
            contributor_patterns=[
                {"role": "author", "title": "Technology Writer"},
                {"role": "author", "title": "Senior Tech Reporter"},
                {"role": "photographer", "title": "Staff Photographer"}
            ]
        )
        
        # News article template
        templates["news_story"] = ArticleTemplate(
            template_name="news_story",
            article_type="news",
            typical_word_count=(400, 1000),
            paragraph_count=(5, 12),
            typical_images=(1, 3),
            title_patterns=[
                "{company} Announces {event}",
                "Breaking: {event} in {location}",
                "{person} {action} at {event}",
                "Local {event} Draws {number} Attendees",
                "{industry} Sees Major {change}"
            ],
            content_themes=[
                "business", "politics", "local events", "economy", "technology",
                "healthcare", "education", "environment"
            ],
            contributor_patterns=[
                {"role": "author", "title": "Staff Reporter"},
                {"role": "author", "title": "News Correspondent"},
                {"role": "photographer", "title": "News Photographer"}
            ]
        )
        
        # Fashion article template
        templates["fashion_feature"] = ArticleTemplate(
            template_name="fashion_feature",
            article_type="feature",
            typical_word_count=(600, 1200),
            paragraph_count=(6, 10),
            typical_images=(3, 6),
            title_patterns=[
                "Spring Trends: {trend} Takes Center Stage",
                "The {adjective} Guide to {item}",
                "{designer}'s Latest Collection",
                "Street Style: {location} Fashion Week",
                "{season} Must-Haves for {audience}"
            ],
            content_themes=[
                "fashion trends", "designer collections", "street style", "accessories",
                "beauty", "lifestyle", "shopping", "style guides"
            ],
            contributor_patterns=[
                {"role": "author", "title": "Fashion Editor"},
                {"role": "author", "title": "Style Writer"},
                {"role": "photographer", "title": "Fashion Photographer"}
            ]
        )
        
        # Lifestyle article template
        templates["lifestyle_piece"] = ArticleTemplate(
            template_name="lifestyle_piece", 
            article_type="lifestyle",
            typical_word_count=(500, 1500),
            paragraph_count=(6, 12),
            typical_images=(2, 5),
            title_patterns=[
                "Living Better: {topic}",
                "The Art of {activity}",
                "{number} Ways to {action}",
                "Discovering {place}: A {adjective} Journey",
                "Why {habit} is Good for You"
            ],
            content_themes=[
                "wellness", "travel", "food", "home design", "relationships",
                "personal growth", "hobbies", "culture"
            ],
            contributor_patterns=[
                {"role": "author", "title": "Lifestyle Writer"},
                {"role": "author", "title": "Contributing Editor"},
                {"role": "photographer", "title": "Lifestyle Photographer"}
            ]
        )
        
        return templates
    
    def _create_media_templates(self) -> Dict[str, MediaTemplate]:
        """Create media templates for different types of images."""
        
        templates = {}
        
        templates["photo"] = MediaTemplate(
            media_type="photo",
            typical_dimensions=(300, 200),
            caption_patterns=[
                "{subject} {action} at {location}",
                "{person} demonstrates {activity}",
                "A view of {location} showing {detail}",
                "{subject} in {setting}",
                "{event} draws {description} crowd"
            ],
            credit_patterns=[
                "Photo by {photographer}",
                "{photographer} for {publication}",
                "Image courtesy of {source}",
                "{photographer}/{publication}"
            ]
        )
        
        templates["illustration"] = MediaTemplate(
            media_type="illustration",
            typical_dimensions=(250, 300),
            caption_patterns=[
                "Illustration showing {concept}",
                "Diagram of {system}",
                "{concept} visualization",
                "How {process} works"
            ],
            credit_patterns=[
                "Illustration by {illustrator}",
                "{illustrator} for {publication}",
                "Graphic: {illustrator}"
            ]
        )
        
        templates["chart"] = MediaTemplate(
            media_type="chart",
            typical_dimensions=(400, 250),
            caption_patterns=[
                "{data} over {timeframe}",
                "Comparison of {metrics}",
                "{statistic} by {category}",
                "Growth in {sector}"
            ],
            credit_patterns=[
                "Chart: {publication} analysis",
                "Data visualization by {analyst}",
                "Source: {source}"
            ]
        )
        
        return templates
    
    def _load_sample_content(self) -> Dict[str, List[str]]:
        """Load sample content for different themes."""
        
        return {
            "tech": [
                "The rapid advancement of artificial intelligence continues to transform industries across the globe.",
                "Machine learning algorithms are becoming increasingly sophisticated in their ability to process complex data.",
                "Cloud computing infrastructure enables businesses to scale their operations more efficiently than ever before.",
                "Cybersecurity threats evolve constantly, requiring continuous innovation in protective technologies.",
                "Data analytics provides unprecedented insights into consumer behavior and market trends.",
                "Blockchain technology promises to revolutionize how we handle secure transactions and data integrity.",
                "The Internet of Things connects everyday devices, creating smart environments in homes and offices.",
                "Quantum computing research advances toward solving previously impossible computational problems."
            ],
            "business": [
                "Market analysts predict significant growth in the renewable energy sector over the next decade.",
                "Corporate sustainability initiatives are becoming essential for long-term business success.",
                "Remote work arrangements continue to reshape traditional office environments and company cultures.",
                "Supply chain disruptions highlight the importance of diversified sourcing strategies.",
                "Digital transformation accelerates as companies adapt to changing consumer expectations.",
                "Investment in employee development shows strong returns in productivity and retention.",
                "Emerging markets present both opportunities and challenges for global expansion.",
                "Financial technology innovations streamline payment processes and improve accessibility."
            ],
            "lifestyle": [
                "Wellness trends emphasize the importance of mental health alongside physical fitness.",
                "Sustainable living practices gain popularity as environmental awareness increases.",
                "Travel experiences focus more on authentic local culture and meaningful connections.",
                "Home design trends reflect the need for flexible, multi-functional living spaces.",
                "Culinary adventures explore diverse flavors and cooking techniques from around the world.",
                "Work-life balance strategies help professionals manage stress and maintain relationships.",
                "Creative hobbies provide outlets for self-expression and stress relief.",
                "Community engagement strengthens social bonds and creates positive local impact."
            ],
            "fashion": [
                "Sustainable fashion gains momentum as consumers become more environmentally conscious.",
                "Vintage and thrift shopping offers unique pieces while reducing fashion waste.",
                "Minimalist wardrobes focus on quality over quantity with versatile, timeless pieces.",
                "Street style influences high fashion designers and mainstream retail trends.",
                "Accessory choices can transform basic outfits into distinctive personal statements.",
                "Seasonal color palettes guide fashion choices and interior design decisions.",
                "Fashion technology integrates smart fabrics and wearable devices into clothing design.",
                "Cultural influences shape contemporary fashion trends and design aesthetics."
            ]
        }
    
    def _load_contributor_names(self) -> Dict[str, List[str]]:
        """Load sample contributor names by role."""
        
        return {
            "first_names": [
                "Sarah", "Michael", "Jennifer", "David", "Lisa", "Robert", "Maria", "James",
                "Jessica", "Christopher", "Ashley", "Matthew", "Amanda", "Daniel", "Emily",
                "Andrew", "Melissa", "Joshua", "Michelle", "Kevin", "Nicole", "Brian", 
                "Angela", "William", "Stephanie", "Thomas", "Rebecca", "John", "Laura"
            ],
            "last_names": [
                "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Wilson",
                "Moore", "Taylor", "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin",
                "Thompson", "Robinson", "Clark", "Rodriguez", "Lewis", "Lee", "Walker", "Hall",
                "Allen", "Young", "King", "Wright", "Lopez", "Hill", "Scott", "Green", "Adams"
            ]
        }
    
    def _load_company_names(self) -> List[str]:
        """Load sample company names for content."""
        
        return [
            "TechCorp", "Innovate Inc", "Digital Solutions", "FutureTech", "DataDrive",
            "CloudWorks", "NextGen Systems", "SmartFlow", "ConnectTech", "GlobalTech",
            "ProSoft", "TechAdvance", "InnovateLab", "DigitalEdge", "TechPioneer",
            "SystemsPlus", "DataTech", "CloudFirst", "TechBridge", "InnovationHub"
        ]
    
    def generate_article(
        self,
        brand_config: BrandConfiguration,
        article_id: Optional[str] = None,
        article_type: Optional[str] = None,
        complexity_level: str = "moderate",
        edge_cases: Optional[List[EdgeCaseType]] = None
    ) -> Dict[str, Any]:
        """
        Generate a complete article with all components.
        
        Args:
            brand_config: Brand configuration
            article_id: Optional specific article ID
            article_type: Optional specific article type
            complexity_level: Content complexity level
            edge_cases: Edge cases to include
            
        Returns:
            Complete article data dictionary
        """
        try:
            self.logger.debug("Generating article",
                            brand=brand_config.brand_name,
                            article_type=article_type,
                            complexity=complexity_level)
            
            # Select article template
            template = self._select_article_template(brand_config, article_type)
            
            # Generate article ID
            if not article_id:
                article_id = f"{brand_config.brand_name.lower()}_{uuid.uuid4().hex[:8]}"
            
            # Generate title
            title = self._generate_title(template, brand_config, edge_cases or [])
            
            # Generate contributors
            contributors = self._generate_contributors(template, edge_cases or [])
            
            # Generate text content
            text_blocks = self._generate_text_content(
                template, brand_config, complexity_level, edge_cases or []
            )
            
            # Generate media elements
            images = self._generate_media_elements(
                template, brand_config, edge_cases or []
            )
            
            # Create article data structure
            article_data = {
                "article_id": article_id,
                "title": title,
                "title_confidence": random.uniform(0.9, 1.0),
                "article_type": template.article_type,
                "contributors": contributors,
                "text_blocks": text_blocks,
                "images": images,
                "generation_timestamp": datetime.now(),
                "template_used": template.template_name,
                "edge_cases": [ec.value for ec in (edge_cases or [])],
                "complexity_level": complexity_level
            }
            
            self.logger.info("Article generated successfully",
                           article_id=article_id,
                           title_length=len(title),
                           contributors=len(contributors),
                           text_blocks=len(text_blocks),
                           images=len(images))
            
            return article_data
            
        except Exception as e:
            self.logger.error("Error generating article", error=str(e))
            raise SyntheticDataError(f"Failed to generate article: {e}")
    
    def _select_article_template(
        self, 
        brand_config: BrandConfiguration,
        article_type: Optional[str]
    ) -> ArticleTemplate:
        """Select appropriate article template."""
        
        if article_type:
            # Find template with matching type
            matching_templates = [
                template for template in self.article_templates.values()
                if template.article_type == article_type
            ]
            if matching_templates:
                return random.choice(matching_templates)
        
        # Select based on brand style
        brand_preferred = {
            BrandStyle.TECH: ["tech_feature", "news_story"],
            BrandStyle.FASHION: ["fashion_feature", "lifestyle_piece"],
            BrandStyle.NEWS: ["news_story", "tech_feature"],
            BrandStyle.LIFESTYLE: ["lifestyle_piece", "fashion_feature"],
            BrandStyle.ACADEMIC: ["tech_feature", "news_story"],
            BrandStyle.TABLOID: ["news_story", "lifestyle_piece"]
        }
        
        preferred_templates = brand_preferred.get(brand_config.brand_style, list(self.article_templates.keys()))
        template_name = random.choice(preferred_templates)
        
        return self.article_templates[template_name]
    
    def _generate_title(
        self, 
        template: ArticleTemplate,
        brand_config: BrandConfiguration,
        edge_cases: List[EdgeCaseType]
    ) -> str:
        """Generate article title."""
        
        # Select title pattern
        title_pattern = random.choice(template.title_patterns)
        
        # Fill in template variables
        theme = random.choice(template.content_themes)
        company = random.choice(self.company_names)
        
        # Create substitution dictionary
        substitutions = {
            "technology": theme,
            "company": company,
            "field": theme,
            "industry": theme.title(),
            "audience": "professionals",
            "event": f"{theme.title()} Conference",
            "location": "Silicon Valley",
            "person": self._generate_person_name(),
            "action": "speaks",
            "number": str(random.randint(100, 5000)),
            "change": "breakthrough",
            "trend": theme,
            "adjective": random.choice(["Ultimate", "Complete", "Essential", "Modern"]),
            "item": theme,
            "designer": company,
            "season": random.choice(["Spring", "Summer", "Fall", "Winter"]),
            "topic": theme,
            "activity": theme.replace("_", " "),
            "place": "New York",
            "habit": theme.replace("_", " ")
        }
        
        # Apply substitutions
        title = title_pattern.format_map(substitutions)
        
        # Handle decorative title edge case
        if EdgeCaseType.DECORATIVE_TITLES in edge_cases:
            # Add decorative elements (simulated in text)
            decorative_prefixes = ["★", "◆", "▲", "●"]
            prefix = random.choice(decorative_prefixes)
            title = f"{prefix} {title} {prefix}"
        
        # Ensure title uniqueness
        base_title = title
        counter = 1
        while title in self._used_titles:
            title = f"{base_title} ({counter})"
            counter += 1
        
        self._used_titles.add(title)
        return title
    
    def _generate_contributors(
        self, 
        template: ArticleTemplate,
        edge_cases: List[EdgeCaseType]
    ) -> List[Dict[str, Any]]:
        """Generate article contributors."""
        
        contributors = []
        
        # Determine number of contributors
        if EdgeCaseType.CONTRIBUTOR_COMPLEXITY in edge_cases:
            contributor_count = random.randint(2, 4)
        else:
            contributor_count = random.randint(1, 2)
        
        for i in range(contributor_count):
            # Select contributor pattern
            if i < len(template.contributor_patterns):
                pattern = template.contributor_patterns[i]
            else:
                pattern = random.choice(template.contributor_patterns)
            
            # Generate name
            name = self._generate_person_name()
            normalized_name = self._normalize_name(name)
            
            # Determine confidence (lower for complex contributors)
            confidence = random.uniform(0.85, 0.98)
            if EdgeCaseType.CONTRIBUTOR_COMPLEXITY in edge_cases:
                confidence *= random.uniform(0.8, 0.95)
            
            contributor = {
                "name": name,
                "normalized_name": normalized_name,
                "role": pattern["role"],
                "title": pattern.get("title", ""),
                "confidence": confidence,
                "extraction_method": "synthetic_generation"
            }
            
            contributors.append(contributor)
        
        return contributors
    
    def _generate_text_content(
        self,
        template: ArticleTemplate,
        brand_config: BrandConfiguration,
        complexity_level: str,
        edge_cases: List[EdgeCaseType]
    ) -> List[Dict[str, Any]]:
        """Generate text content blocks."""
        
        text_blocks = []
        
        # Determine paragraph count
        min_paragraphs, max_paragraphs = template.paragraph_count
        
        if complexity_level == "simple":
            paragraph_count = random.randint(min_paragraphs, (min_paragraphs + max_paragraphs) // 2)
        elif complexity_level == "complex":
            paragraph_count = random.randint((min_paragraphs + max_paragraphs) // 2, max_paragraphs)
        else:  # moderate
            paragraph_count = random.randint(min_paragraphs, max_paragraphs)
        
        # Get content theme
        content_pool = self.sample_content.get(
            template.content_themes[0].split()[0], 
            self.sample_content["business"]
        )
        
        for i in range(paragraph_count):
            # Select paragraph type
            if i == 0:
                block_type = "paragraph"
            elif i == paragraph_count - 1:
                block_type = "paragraph"
            elif random.random() < 0.1:  # 10% chance of pullquote
                block_type = "pullquote"
            else:
                block_type = "paragraph"
            
            # Generate text content
            if block_type == "pullquote":
                # Shorter, impactful text for pullquotes
                text = random.choice(content_pool)
            else:
                # Longer paragraph text
                sentences = random.sample(content_pool, random.randint(2, 4))
                text = " ".join(sentences)
            
            # Determine confidence
            confidence = random.uniform(0.88, 0.99)
            if block_type == "pullquote":
                confidence *= 0.95  # Pullquotes slightly harder to extract
            
            text_block = {
                "id": f"block_{i+1:03d}",
                "type": block_type,
                "text": text,
                "confidence": confidence,
                "word_count": len(text.split()),
                "position": i + 1,
                "reading_order": i + 10  # Leave room for title/byline
            }
            
            text_blocks.append(text_block)
        
        return text_blocks
    
    def _generate_media_elements(
        self,
        template: ArticleTemplate,
        brand_config: BrandConfiguration,
        edge_cases: List[EdgeCaseType]
    ) -> List[Dict[str, Any]]:
        """Generate media elements (images, etc.)."""
        
        images = []
        
        # Determine number of images
        min_images, max_images = template.typical_images
        image_count = random.randint(min_images, max_images)
        
        # Adjust for edge cases
        if EdgeCaseType.CAPTION_AMBIGUITY in edge_cases:
            image_count = max(2, image_count)  # Need multiple images for ambiguity
        
        for i in range(image_count):
            # Select media template
            media_template = random.choice(list(self.media_templates.values()))
            
            # Generate filename
            filename = f"img_{i+1:03d}_{template.template_name}.jpg"
            
            # Generate caption
            caption = self._generate_caption(media_template, template)
            
            # Generate credit
            credit = self._generate_credit(media_template)
            
            # Determine confidence
            confidence = random.uniform(0.85, 0.97)
            if EdgeCaseType.CAPTION_AMBIGUITY in edge_cases:
                confidence *= random.uniform(0.7, 0.9)  # Lower confidence for ambiguous cases
            
            image_data = {
                "filename": filename,
                "caption": caption,
                "credit": credit,
                "confidence": confidence,
                "media_type": media_template.media_type,
                "dimensions": media_template.typical_dimensions,
                "alt_text": caption[:100] + "..." if len(caption) > 100 else caption
            }
            
            images.append(image_data)
        
        return images
    
    def _generate_caption(self, media_template: MediaTemplate, article_template: ArticleTemplate) -> str:
        """Generate image caption."""
        
        caption_pattern = random.choice(media_template.caption_patterns)
        
        # Fill in caption variables
        theme = random.choice(article_template.content_themes)
        
        substitutions = {
            "subject": theme.title(),
            "action": "demonstrates",
            "location": "headquarters",
            "person": self._generate_person_name(),
            "activity": theme.replace("_", " "),
            "detail": "latest developments",
            "setting": "conference room",
            "event": f"{theme.title()} Summit",
            "description": "enthusiastic",
            "concept": theme,
            "system": f"{theme} framework",
            "process": theme.replace("_", " "),
            "data": f"{theme.title()} metrics",
            "timeframe": "2020-2024",
            "metrics": "performance indicators",
            "statistic": "growth rates",
            "category": "industry sector",
            "sector": theme
        }
        
        return caption_pattern.format_map(substitutions)
    
    def _generate_credit(self, media_template: MediaTemplate) -> str:
        """Generate media credit line."""
        
        credit_pattern = random.choice(media_template.credit_patterns)
        
        # Generate photographer/illustrator name
        if media_template.media_type == "illustration":
            creator_name = self._generate_person_name()
            role = "illustrator"
        else:
            creator_name = self._generate_person_name()
            role = "photographer"
        
        substitutions = {
            "photographer": creator_name,
            "illustrator": creator_name,
            "publication": "Stock Photos",
            "source": "Getty Images",
            "analyst": creator_name
        }
        
        return credit_pattern.format_map(substitutions)
    
    def _generate_person_name(self) -> str:
        """Generate realistic person name."""
        
        first_name = random.choice(self.contributor_names["first_names"])
        last_name = random.choice(self.contributor_names["last_names"])
        
        return f"{first_name} {last_name}"
    
    def _normalize_name(self, name: str) -> str:
        """Convert name to Last, First format."""
        
        parts = name.split()
        if len(parts) >= 2:
            first = parts[0]
            last = parts[-1]
            middle = " ".join(parts[1:-1]) if len(parts) > 2 else ""
            
            if middle:
                return f"{last}, {first} {middle}"
            else:
                return f"{last}, {first}"
        else:
            return name
    
    def get_content_statistics(self) -> Dict[str, Any]:
        """Get content generation statistics."""
        
        return {
            "article_templates": len(self.article_templates),
            "media_templates": len(self.media_templates),
            "sample_content_themes": list(self.sample_content.keys()),
            "contributor_names_available": (
                len(self.contributor_names["first_names"]) * 
                len(self.contributor_names["last_names"])
            ),
            "titles_generated": len(self._used_titles),
            "company_names": len(self.company_names)
        }