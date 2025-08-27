"""
Common types used across multiple modules to avoid circular imports.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class BoundingBox:
    """Represents a bounding box for layout elements."""

    x: float
    y: float
    width: float
    height: float

    @property
    def x2(self) -> float:
        """Right edge of the bounding box."""
        return self.x + self.width

    @property
    def y2(self) -> float:
        """Bottom edge of the bounding box."""
        return self.y + self.height

    @property
    def center_x(self) -> float:
        """X coordinate of the center."""
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        """Y coordinate of the center."""
        return self.y + self.height / 2

    @property
    def area(self) -> float:
        """Area of the bounding box."""
        return self.width * self.height

    def overlaps(self, other: "BoundingBox") -> bool:
        """Check if this bounding box overlaps with another."""
        return not (
            self.x2 < other.x
            or other.x2 < self.x
            or self.y2 < other.y
            or other.y2 < self.y
        )

    def contains(self, other: "BoundingBox") -> bool:
        """Check if this bounding box contains another."""
        return (
            self.x <= other.x
            and self.y <= other.y
            and self.x2 >= other.x2
            and self.y2 >= other.y2
        )

    def intersection(self, other: "BoundingBox") -> "BoundingBox":
        """Calculate the intersection with another bounding box."""
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)

        if x2 > x1 and y2 > y1:
            return BoundingBox(x1, y1, x2 - x1, y2 - y1)
        else:
            return BoundingBox(0, 0, 0, 0)

    def union(self, other: "BoundingBox") -> "BoundingBox":
        """Calculate the union with another bounding box."""
        x1 = min(self.x, other.x)
        y1 = min(self.y, other.y)
        x2 = max(self.x2, other.x2)
        y2 = max(self.y2, other.y2)

        return BoundingBox(x1, y1, x2 - x1, y2 - y1)

    def to_tuple(self) -> Tuple[float, float, float, float]:
        """Convert to tuple (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)
