"""Temporal entity extraction and resolution.

Extracts time expressions from text and resolves relative dates
(e.g., "yesterday", "last week") to absolute dates using a reference timestamp.

Uses dateparser for robust temporal parsing across multiple formats and languages.
"""

import re
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import dateparser, fall back gracefully if not available
try:
    import dateparser
    DATEPARSER_AVAILABLE = True
except ImportError:
    DATEPARSER_AVAILABLE = False
    logger.warning("dateparser not installed. Temporal extraction will be limited.")


@dataclass
class TemporalEntity:
    """An extracted and resolved temporal expression."""
    
    expression: str  # Original text: "yesterday", "May 7, 2023"
    resolved_date: Optional[str]  # ISO format: "2023-05-07"
    precision: str  # "day", "week", "month", "year", "time"
    reference_date: str  # ISO format of the reference timestamp
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)


@dataclass 
class TemporalRelationship:
    """A temporal relationship between an event/entity and a time."""
    
    subject: str  # Entity or event name
    relation_type: str  # "occurred_on", "mentioned_date", "deadline", etc.
    temporal: TemporalEntity


class TemporalExtractor:
    """Extract and resolve temporal expressions from text.
    
    Handles:
    - Relative expressions: "yesterday", "last week", "3 days ago"
    - Absolute dates: "May 7, 2023", "2023-05-07", "May 2023"
    - Named days: "Monday", "next Tuesday"
    - Durations: "for 2 weeks", "over 3 months"
    
    All relative expressions are resolved using a reference timestamp.
    """
    
    # Patterns for temporal expressions we want to capture
    RELATIVE_PATTERNS: list[str] = [
        # Days
        r'\byesterday\b',
        r'\btoday\b',
        r'\btomorrow\b',
        r'\b(?:last|this|next)\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
        r'\b\d+\s+days?\s+ago\b',
        r'\bin\s+\d+\s+days?\b',
        
        # Weeks
        r'\blast\s+week\b',
        r'\bthis\s+week\b',
        r'\bnext\s+week\b',
        r'\b\d+\s+weeks?\s+ago\b',
        
        # Months
        r'\blast\s+month\b',
        r'\bthis\s+month\b',
        r'\bnext\s+month\b',
        r'\b\d+\s+months?\s+ago\b',
        
        # Years
        r'\blast\s+year\b',
        r'\bthis\s+year\b',
        r'\bnext\s+year\b',
        r'\b\d+\s+years?\s+ago\b',
        
        # Specific times
        r'\bthis\s+morning\b',
        r'\bthis\s+afternoon\b',
        r'\bthis\s+evening\b',
        r'\blast\s+night\b',
        r'\btonight\b',
    ]
    
    # Month names for reuse in patterns
    _MONTHS = (
        r'january|february|march|april|may|june'
        r'|july|august|september|october|november|december'
    )

    # Patterns for absolute dates
    ABSOLUTE_PATTERNS: list[str] = [
        # ISO format
        r'\b\d{4}-\d{2}-\d{2}\b',
        # US format
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
        # "Month Day, Year"
        r'\b(?:' + _MONTHS + r')'
        r'\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{4}\b',
        # "Day Month, Year"
        r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:' + _MONTHS + r')'
        r',?\s*\d{4}\b',
        # "Month Year"
        r'\b(?:' + _MONTHS + r')\s+\d{4}\b',
        # Just year (4 digits, likely a year)
        r'\bin\s+\d{4}\b',
    ]
    
    def __init__(self):
        """Initialize the temporal extractor."""
        self._relative_regex = re.compile(
            '|'.join(f'({p})' for p in self.RELATIVE_PATTERNS),
            re.IGNORECASE
        )
        self._absolute_regex = re.compile(
            '|'.join(f'({p})' for p in self.ABSOLUTE_PATTERNS),
            re.IGNORECASE
        )
    
    def extract(
        self, 
        text: str, 
        reference_time: Optional[datetime] = None
    ) -> list[TemporalEntity]:
        """Extract temporal expressions from text.
        
        Args:
            text: Input text to extract from.
            reference_time: Reference datetime for resolving relative expressions.
                           Defaults to UTC now.
        
        Returns:
            List of extracted and resolved TemporalEntity objects.
        """
        if not text or not text.strip():
            return []
        
        if reference_time is None:
            reference_time = datetime.now(timezone.utc)
        
        reference_iso = reference_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        entities = []
        seen_expressions: set[str] = set()
        
        # Extract relative expressions
        for match in self._relative_regex.finditer(text):
            expr = match.group(0).strip()
            if expr.lower() in seen_expressions:
                continue
            seen_expressions.add(expr.lower())
            
            resolved = self._resolve_expression(expr, reference_time)
            if resolved:
                entities.append(TemporalEntity(
                    expression=expr,
                    resolved_date=resolved['date'],
                    precision=resolved['precision'],
                    reference_date=reference_iso,
                    confidence=resolved.get('confidence', 0.9),
                ))
        
        # Extract absolute expressions
        for match in self._absolute_regex.finditer(text):
            expr = match.group(0).strip()
            if expr.lower() in seen_expressions:
                continue
            seen_expressions.add(expr.lower())
            
            resolved = self._resolve_expression(expr, reference_time)
            if resolved:
                entities.append(TemporalEntity(
                    expression=expr,
                    resolved_date=resolved['date'],
                    precision=resolved['precision'],
                    reference_date=reference_iso,
                    confidence=resolved.get('confidence', 1.0),
                ))
        
        return entities
    
    def _resolve_expression(
        self, 
        expression: str, 
        reference_time: datetime
    ) -> Optional[dict]:
        """Resolve a temporal expression to an absolute date.
        
        Args:
            expression: Temporal expression to resolve.
            reference_time: Reference datetime for relative expressions.
        
        Returns:
            Dict with 'date' (ISO string), 'precision', and 'confidence'.
        """
        if not DATEPARSER_AVAILABLE:
            return self._resolve_fallback(expression, reference_time)
        
        try:
            # Configure dateparser for relative date handling
            settings = {
                'RELATIVE_BASE': reference_time,
                'PREFER_DATES_FROM': 'past',
                'RETURN_AS_TIMEZONE_AWARE': True,
            }
            
            parsed = dateparser.parse(expression, settings=settings)
            
            if parsed:
                # Determine precision based on expression
                precision = self._infer_precision(expression)
                
                # Format date based on precision
                if precision == 'year':
                    date_str = str(parsed.year)
                elif precision == 'month':
                    date_str = parsed.strftime("%Y-%m")
                else:
                    date_str = parsed.strftime("%Y-%m-%d")
                
                return {
                    'date': date_str,
                    'precision': precision,
                    'confidence': 0.95 if precision == 'day' else 0.8,
                }
        except (ValueError, TypeError, AttributeError, OverflowError) as e:
            logger.debug(
                "dateparser failed for '%s': %s", expression, e
            )
        
        return self._resolve_fallback(expression, reference_time)
    
    def _resolve_fallback(
        self, 
        expression: str, 
        reference_time: datetime
    ) -> Optional[dict]:
        """Fallback resolution for common expressions without dateparser."""
        expr_lower = expression.lower().strip()
        
        from datetime import timedelta
        
        if expr_lower == 'yesterday':
            date = reference_time - timedelta(days=1)
            return {'date': date.strftime("%Y-%m-%d"), 'precision': 'day', 'confidence': 1.0}
        
        if expr_lower == 'today':
            return {'date': reference_time.strftime("%Y-%m-%d"), 'precision': 'day', 'confidence': 1.0}
        
        if expr_lower == 'tomorrow':
            date = reference_time + timedelta(days=1)
            return {'date': date.strftime("%Y-%m-%d"), 'precision': 'day', 'confidence': 1.0}
        
        # N days ago
        days_ago_match = re.match(r'(\d+)\s+days?\s+ago', expr_lower)
        if days_ago_match:
            days = int(days_ago_match.group(1))
            date = reference_time - timedelta(days=days)
            return {'date': date.strftime("%Y-%m-%d"), 'precision': 'day', 'confidence': 0.95}
        
        # N weeks ago
        weeks_ago_match = re.match(r'(\d+)\s+weeks?\s+ago', expr_lower)
        if weeks_ago_match:
            weeks = int(weeks_ago_match.group(1))
            date = reference_time - timedelta(weeks=weeks)
            return {'date': date.strftime("%Y-%m-%d"), 'precision': 'week', 'confidence': 0.9}
        
        # Last week/month/year
        if expr_lower == 'last week':
            date = reference_time - timedelta(weeks=1)
            return {'date': date.strftime("%Y-%m-%d"), 'precision': 'week', 'confidence': 0.85}
        
        if expr_lower == 'last month':
            # Use calendar-aware month subtraction
            month = reference_time.month - 1
            year = reference_time.year
            if month < 1:
                month = 12
                year -= 1
            return {
                'date': f"{year}-{month:02d}",
                'precision': 'month',
                'confidence': 0.8,
            }
        
        if expr_lower == 'last year':
            year = reference_time.year - 1
            return {'date': str(year), 'precision': 'year', 'confidence': 0.8}
        
        # Year only (e.g., "in 2022")
        year_match = re.match(r'in\s+(\d{4})', expr_lower)
        if year_match:
            return {'date': year_match.group(1), 'precision': 'year', 'confidence': 1.0}
        
        return None
    
    def _infer_precision(self, expression: str) -> str:
        """Infer the precision of a temporal expression."""
        expr_lower = expression.lower().strip()
        
        # Year patterns
        if 'year' in expr_lower:
            return 'year'
        # "in 2022" pattern
        if re.match(r'^in\s+\d{4}$', expr_lower):
            return 'year'
        # Just a year "2022"
        if re.match(r'^\d{4}$', expr_lower):
            return 'year'
        
        # Week patterns
        if 'week' in expr_lower:
            return 'week'
        
        # Month patterns
        if 'month' in expr_lower:
            return 'month'
        
        # Day patterns
        if any(x in expr_lower for x in ['yesterday', 'today', 'tomorrow', 'day']):
            return 'day'
        
        # Time of day
        if any(x in expr_lower for x in ['morning', 'afternoon', 'evening', 'night', ':']):
            return 'time'
        
        # Default to day for specific dates
        return 'day'
    
    def extract_with_context(
        self,
        text: str,
        reference_time: Optional[datetime] = None,
        entities: Optional[list] = None,
    ) -> list[TemporalRelationship]:
        """Extract temporal expressions with contextual relationships.
        
        Attempts to link temporal expressions to nearby entities/events
        mentioned in the text.
        
        Args:
            text: Input text.
            reference_time: Reference datetime for resolution.
            entities: Optional list of already-extracted entities for linking.
        
        Returns:
            List of TemporalRelationship objects.
        """
        temporal_entities = self.extract(text, reference_time)
        relationships = []
        
        for temp in temporal_entities:
            # Find what the temporal expression relates to
            # Look for nearby verbs/events
            subject = self._find_temporal_subject(text, temp.expression)
            
            relationships.append(TemporalRelationship(
                subject=subject or "event",
                relation_type="occurred_on" if subject else "mentioned_date",
                temporal=temp,
            ))
        
        return relationships
    
    def _find_temporal_subject(self, text: str, expression: str) -> Optional[str]:
        """Find what a temporal expression refers to in the text.
        
        Looks for patterns like "I [verb] [expression]" or 
        "[subject] [expression]".
        """
        # Common patterns: "I went/attended/visited ... yesterday"
        patterns = [
            rf'I\s+(\w+ed)\s+(?:to\s+)?(?:a\s+|an\s+|the\s+)?(.+?)\s+{re.escape(expression)}',
            rf'(\w+)\s+{re.escape(expression)}',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Return the verb or subject found
                groups = [g for g in match.groups() if g]
                if groups:
                    return groups[-1].strip()
        
        return None


def format_temporal_context(
    memory_content: str,
    memory_timestamp: datetime,
    temporal_entities: list[TemporalEntity],
) -> str:
    """Format temporal context for inclusion in LLM prompts.
    
    Args:
        memory_content: Original memory text.
        memory_timestamp: When the memory was created.
        temporal_entities: Extracted temporal entities.
    
    Returns:
        Formatted context string with resolved dates.
    """
    lines = [
        f"[Memory from {memory_timestamp.strftime('%B %d, %Y')}]",
        memory_content,
    ]
    
    if temporal_entities:
        resolved = []
        for temp in temporal_entities:
            if temp.resolved_date and temp.expression.lower() != temp.resolved_date:
                resolved.append(f'"{temp.expression}" → {temp.resolved_date}')
        
        if resolved:
            lines.append(f"[Resolved dates: {'; '.join(resolved)}]")
    
    return '\n'.join(lines)
