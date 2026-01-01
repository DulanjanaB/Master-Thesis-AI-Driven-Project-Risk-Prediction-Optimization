"""
LessonsLearnedParser: Extracts insights from unstructured Lessons Learned documents.
"""

import re
from typing import List, Dict, Optional
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class LessonsLearnedParser:
    """
    Parses and extracts structured information from Lessons Learned documents.
    
    Extracts:
    - Risk keywords and themes
    - Common issues and challenges
    - Success factors
    - Recommendations
    """
    
    def __init__(self):
        """Initialize parser with keyword dictionaries."""
        # Risk-related keywords
        self.risk_keywords = {
            'schedule': ['delay', 'timeline', 'deadline', 'schedule', 'overdue', 'late'],
            'budget': ['cost', 'budget', 'overrun', 'expense', 'financial'],
            'technical': ['technical', 'technology', 'integration', 'compatibility', 'bug'],
            'resource': ['resource', 'staffing', 'team', 'shortage', 'turnover'],
            'scope': ['scope', 'requirement', 'change', 'creep', 'expansion'],
            'quality': ['quality', 'defect', 'error', 'testing', 'reliability'],
            'communication': ['communication', 'coordination', 'stakeholder', 'alignment']
        }
        
        # Issue indicators
        self.issue_patterns = [
            r'problem[s]?\s+(?:with|in|regarding)',
            r'challenge[s]?\s+(?:with|in|faced)',
            r'issue[s]?\s+(?:with|in|regarding)',
            r'difficult(?:y|ies)\s+(?:with|in)',
            r'obstacle[s]?\s+(?:in|to)',
            r'failed?\s+to',
            r'unable\s+to',
            r'lack\s+of'
        ]
        
        # Success indicators
        self.success_patterns = [
            r'success(?:ful|fully)',
            r'achieved?',
            r'accomplished?',
            r'delivered?\s+(?:on\s+time|successfully)',
            r'worked?\s+well',
            r'effective(?:ly)?'
        ]
    
    def parse_document(self, content: str) -> Dict:
        """
        Parse a single Lessons Learned document.
        
        Args:
            content: Document content as string
            
        Returns:
            Dictionary with extracted information
        """
        content_lower = content.lower()
        
        # Extract risk themes
        risk_themes = self._extract_risk_themes(content_lower)
        
        # Identify issues and challenges
        issues = self._extract_issues(content)
        
        # Identify success factors
        successes = self._extract_successes(content)
        
        # Calculate sentiment
        sentiment_score = self._calculate_sentiment(content_lower, issues, successes)
        
        return {
            'risk_themes': risk_themes,
            'issues_mentioned': len(issues),
            'successes_mentioned': len(successes),
            'sentiment_score': sentiment_score,
            'dominant_risk': max(risk_themes, key=risk_themes.get) if risk_themes else None,
            'issues': issues[:5],  # Top 5 issues
            'successes': successes[:5]  # Top 5 successes
        }
    
    def _extract_risk_themes(self, content: str) -> Dict[str, int]:
        """
        Extract risk themes from content.
        
        Args:
            content: Document content (lowercase)
            
        Returns:
            Dictionary mapping risk categories to frequency
        """
        risk_counts = {}
        
        for category, keywords in self.risk_keywords.items():
            count = sum(content.count(keyword) for keyword in keywords)
            if count > 0:
                risk_counts[category] = count
        
        return risk_counts
    
    def _extract_issues(self, content: str) -> List[str]:
        """
        Extract issue mentions from content.
        
        Args:
            content: Document content
            
        Returns:
            List of issue sentences
        """
        issues = []
        sentences = re.split(r'[.!?]+', content)
        
        for sentence in sentences:
            for pattern in self.issue_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    issues.append(sentence.strip())
                    break
        
        return issues
    
    def _extract_successes(self, content: str) -> List[str]:
        """
        Extract success mentions from content.
        
        Args:
            content: Document content
            
        Returns:
            List of success sentences
        """
        successes = []
        sentences = re.split(r'[.!?]+', content)
        
        for sentence in sentences:
            for pattern in self.success_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    successes.append(sentence.strip())
                    break
        
        return successes
    
    def _calculate_sentiment(self, content: str, issues: List[str], successes: List[str]) -> float:
        """
        Calculate document sentiment score.
        
        Args:
            content: Document content (lowercase)
            issues: List of issue mentions
            successes: List of success mentions
            
        Returns:
            Sentiment score between -1 (negative) and 1 (positive)
        """
        # Simple sentiment based on positive/negative word counts
        negative_words = ['failed', 'problem', 'issue', 'delay', 'overrun', 'challenge', 
                         'difficult', 'poor', 'unsuccessful', 'missed']
        positive_words = ['success', 'achieved', 'delivered', 'effective', 'improved',
                         'excellent', 'completed', 'accomplished', 'well']
        
        neg_count = sum(content.count(word) for word in negative_words)
        pos_count = sum(content.count(word) for word in positive_words)
        
        total = neg_count + pos_count
        if total == 0:
            return 0.0
        
        return (pos_count - neg_count) / total
    
    def parse_multiple(self, documents: List[Dict]) -> Dict:
        """
        Parse multiple Lessons Learned documents and aggregate insights.
        
        Args:
            documents: List of document dictionaries with 'content' field
            
        Returns:
            Aggregated insights across all documents
        """
        all_risks = Counter()
        all_issues = []
        all_successes = []
        sentiment_scores = []
        
        for doc in documents:
            content = doc.get('content', '')
            if not content:
                continue
            
            parsed = self.parse_document(content)
            
            # Aggregate risk themes
            all_risks.update(parsed['risk_themes'])
            
            # Collect issues and successes
            all_issues.extend(parsed['issues'])
            all_successes.extend(parsed['successes'])
            
            # Collect sentiment
            sentiment_scores.append(parsed['sentiment_score'])
        
        return {
            'total_documents': len(documents),
            'risk_theme_summary': dict(all_risks.most_common()),
            'top_risk_categories': [cat for cat, _ in all_risks.most_common(3)],
            'total_issues_found': len(all_issues),
            'total_successes_found': len(all_successes),
            'average_sentiment': sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0,
            'sample_issues': all_issues[:10],
            'sample_successes': all_successes[:10]
        }
    
    def generate_risk_features(self, content: str) -> Dict[str, float]:
        """
        Generate numerical risk features from document content.
        
        Args:
            content: Document content
            
        Returns:
            Dictionary of risk features
        """
        parsed = self.parse_document(content)
        content_lower = content.lower()
        
        features = {
            'll_schedule_risk': parsed['risk_themes'].get('schedule', 0),
            'll_budget_risk': parsed['risk_themes'].get('budget', 0),
            'll_technical_risk': parsed['risk_themes'].get('technical', 0),
            'll_resource_risk': parsed['risk_themes'].get('resource', 0),
            'll_scope_risk': parsed['risk_themes'].get('scope', 0),
            'll_quality_risk': parsed['risk_themes'].get('quality', 0),
            'll_communication_risk': parsed['risk_themes'].get('communication', 0),
            'll_sentiment': parsed['sentiment_score'],
            'll_issue_count': parsed['issues_mentioned'],
            'll_success_count': parsed['successes_mentioned'],
            'll_document_length': len(content.split())
        }
        
        return features
