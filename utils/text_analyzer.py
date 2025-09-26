#!/usr/bin/env python3
"""
Text Analyzer for SAP Incident Analysis
Provides text processing and keyword analysis capabilities.
"""

import re
import pandas as pd
from typing import Dict, List, Set, Any, Tuple
from collections import Counter
import logging

class TextAnalyzer:
    """
    Text analysis utility following Python best practices.
    Implements proper string operations and mathematical calculations.
    """
    
    def __init__(self):
        """Initialize the Text Analyzer."""
        self.logger = logging.getLogger(__name__)
        self.sap_keywords = self._load_sap_keywords()
        self.priority_keywords = self._load_priority_keywords()
    
    def _load_sap_keywords(self) -> Dict[str, List[str]]:
        """Load SAP-specific keywords for classification."""
        return {
            'basis': ['basis', 'system', 'server', 'database', 'performance', 'memory'],
            'security': ['security', 'authorization', 'user', 'role', 'access', 'permission'],
            'interface': ['interface', 'idoc', 'rfc', 'bapi', 'webservice', 'integration'],
            'functional': ['functional', 'business', 'process', 'workflow', 'configuration'],
            'development': ['development', 'abap', 'custom', 'enhancement', 'modification'],
            'data': ['data', 'master', 'transaction', 'posting', 'migration', 'archiving']
        }
    
    def _load_priority_keywords(self) -> Dict[str, List[str]]:
        """Load priority-indicating keywords."""
        return {
            'critical': ['critical', 'urgent', 'emergency', 'down', 'outage', 'production'],
            'high': ['high', 'important', 'asap', 'priority', 'escalate', 'business'],
            'medium': ['medium', 'normal', 'standard', 'regular'],
            'low': ['low', 'minor', 'question', 'request', 'information', 'training']
        }
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Perform comprehensive text analysis using string operations.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            if not text or pd.isna(text):
                return self._empty_analysis()
            
            # Clean and normalize text
            cleaned_text = self._clean_text(str(text))
            
            # Calculate basic metrics using mathematical operations
            word_count = len(cleaned_text.split())
            char_count = len(cleaned_text)
            
            # Find SAP keywords
            sap_matches = self._find_sap_keywords(cleaned_text)
            
            # Analyze priority indicators
            priority_analysis = self._analyze_priority_indicators(cleaned_text)
            
            # Calculate keyword density using division
            keyword_density = len(sap_matches) / word_count if word_count > 0 else 0
            
            return {
                'word_count': word_count,
                'char_count': char_count,
                'sap_keywords': sap_matches,
                'priority_indicators': priority_analysis,
                'keyword_density': round(keyword_density * 100, 2),  # Using multiplication
                'suggested_category': self._suggest_category(sap_matches),
                'urgency_score': self._calculate_urgency_score(priority_analysis)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing text: {e}")
            return self._empty_analysis()
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for analysis."""
        try:
            # Convert to lowercase
            cleaned = text.lower()
            
            # Remove special characters but keep spaces
            cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
            
            # Remove extra whitespace
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            
            return cleaned
            
        except Exception:
            return ""
    
    def _find_sap_keywords(self, text: str) -> Dict[str, List[str]]:
        """Find SAP-specific keywords in text."""
        found_keywords = {}
        
        try:
            for category, keywords in self.sap_keywords.items():
                matches = []
                for keyword in keywords:
                    # Use string operations to find keywords
                    if keyword in text:
                        matches.append(keyword)
                
                if matches:
                    found_keywords[category] = matches
            
            return found_keywords
            
        except Exception:
            return {}
    
    def _analyze_priority_indicators(self, text: str) -> Dict[str, Any]:
        """Analyze priority-indicating keywords using mathematical operations."""
        try:
            priority_scores = {}
            total_matches = 0
            
            # Count matches for each priority level using addition
            for priority, keywords in self.priority_keywords.items():
                matches = sum(1 for keyword in keywords if keyword in text)
                priority_scores[priority] = matches
                total_matches += matches  # Addition operation
            
            # Calculate percentages using division and multiplication
            priority_percentages = {}
            if total_matches > 0:
                for priority, count in priority_scores.items():
                    percentage = (count / total_matches) * 100
                    priority_percentages[priority] = round(percentage, 1)
            else:
                priority_percentages = {priority: 0.0 for priority in priority_scores.keys()}
            
            return {
                'raw_counts': priority_scores,
                'percentages': priority_percentages,
                'total_indicators': total_matches
            }
            
        except Exception:
            return {'raw_counts': {}, 'percentages': {}, 'total_indicators': 0}
    
    def _suggest_category(self, sap_matches: Dict[str, List[str]]) -> str:
        """Suggest SAP category based on keyword matches."""
        if not sap_matches:
            return 'general'
        
        # Find category with most matches using mathematical comparison
        max_matches = 0
        suggested_category = 'general'
        
        for category, keywords in sap_matches.items():
            match_count = len(keywords)
            if match_count > max_matches:  # Comparison operation
                max_matches = match_count
                suggested_category = category
        
        return suggested_category
    
    def _calculate_urgency_score(self, priority_analysis: Dict[str, Any]) -> float:
        """Calculate urgency score using mathematical operations."""
        try:
            raw_counts = priority_analysis.get('raw_counts', {})
            
            # Weight different priority levels using multiplication
            weights = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
            
            total_score = 0
            total_weight = 0
            
            for priority, count in raw_counts.items():
                weight = weights.get(priority, 1)
                total_score += count * weight  # Multiplication and addition
                total_weight += count  # Addition
            
            # Calculate weighted average using division
            if total_weight > 0:
                urgency_score = total_score / total_weight
            else:
                urgency_score = 0
            
            return round(urgency_score, 2)
            
        except Exception:
            return 0.0
    
    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis structure."""
        return {
            'word_count': 0,
            'char_count': 0,
            'sap_keywords': {},
            'priority_indicators': {'raw_counts': {}, 'percentages': {}, 'total_indicators': 0},
            'keyword_density': 0.0,
            'suggested_category': 'general',
            'urgency_score': 0.0
        }
    
    def batch_analyze(self, text_series: pd.Series) -> pd.DataFrame:
        """
        Perform batch text analysis on a pandas Series.
        
        Args:
            text_series: Series containing text to analyze
            
        Returns:
            DataFrame with analysis results
        """
        try:
            results = []
            
            # Process each text using a loop
            for idx, text in text_series.items():
                analysis = self.analyze_text(text)
                analysis['index'] = idx
                results.append(analysis)
            
            return pd.DataFrame(results)
            
        except Exception as e:
            self.logger.error(f"Error in batch analysis: {e}")
            return pd.DataFrame()
    
    def extract_common_phrases(self, text_series: pd.Series, min_length: int = 3) -> List[Tuple[str, int]]:
        """
        Extract common phrases from text series using mathematical operations.
        
        Args:
            text_series: Series containing text
            min_length: Minimum phrase length in words
            
        Returns:
            List of (phrase, frequency) tuples
        """
        try:
            all_phrases = []
            
            for text in text_series.dropna():
                cleaned = self._clean_text(str(text))
                words = cleaned.split()
                
                # Generate phrases of specified length using loops
                for i in range(len(words) - min_length + 1):
                    phrase = ' '.join(words[i:i + min_length])
                    all_phrases.append(phrase)
            
            # Count phrase frequencies using Counter
            phrase_counts = Counter(all_phrases)
            
            # Filter phrases that appear more than once using comparison
            common_phrases = [(phrase, count) for phrase, count in phrase_counts.items() if count > 1]
            
            # Sort by frequency using comparison operations
            common_phrases.sort(key=lambda x: x[1], reverse=True)
            
            return common_phrases[:20]  # Return top 20
            
        except Exception as e:
            self.logger.error(f"Error extracting phrases: {e}")
            return []