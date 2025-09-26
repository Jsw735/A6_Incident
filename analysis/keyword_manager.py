#!/usr/bin/env python3
"""
Dynamic Keyword Manager for SAP Incident Analysis
Learns and manages SAP-related keywords for classification.
"""

import pandas as pd
import json
import re
from typing import Dict, List, Set, Any
from collections import Counter
import logging
from datetime import datetime  # Add this line!



class IncidentKeywordManager:
    """
    Manages dynamic learning and classification of SAP keywords.
    Implements self-learning capabilities for better incident classification.
    """
    
    def __init__(self, seed_keywords_file: str = None):
        """Initialize the Keyword Manager with seed keywords."""
        self.logger = logging.getLogger(__name__)
        self.sap_keywords = set()
        self.keyword_scores = {}
        self.classification_rules = {}
        
        # Load seed keywords if provided
        if seed_keywords_file:
            self._load_seed_keywords(seed_keywords_file)
        else:
            self._initialize_default_keywords()
    
    def _load_seed_keywords(self, file_path: str):
        """Load initial SAP keywords from JSON file."""
        try:
            with open(file_path, 'r') as f:
                seed_data = json.load(f)
                self.sap_keywords = set(seed_data.get('keywords', []))
                self.classification_rules = seed_data.get('rules', {})
        except FileNotFoundError:
            self.logger.warning(f"Seed file {file_path} not found, using defaults")
            self._initialize_default_keywords()
        except Exception as e:
            self.logger.error(f"Error loading seed keywords: {e}")
            self._initialize_default_keywords()
    
    def _initialize_default_keywords(self):
        """Initialize with default SAP keywords."""
        self.sap_keywords = {
            # SAP Modules
            'sap', 'abap', 'basis', 'hana', 'fiori', 'ui5', 'bw', 'bi',
            'mm', 'sd', 'fi', 'co', 'hr', 'pp', 'pm', 'qm', 'ps',
            
            # Technical Terms
            'rfc', 'bapi', 'idoc', 'ale', 'edi', 'workflow', 'smartforms',
            'sapscript', 'enhancement', 'badi', 'user-exit', 'transaction',
            'tcode', 'table', 'function', 'class', 'method',
            
            # System Terms
            'client', 'mandant', 'transport', 'customizing', 'configuration',
            'authorization', 'role', 'profile', 'user', 'lock', 'update',
            'batch', 'background', 'spool', 'variant', 'selection-screen'
        }
        
        self.classification_rules = {
            'technical': ['abap', 'basis', 'rfc', 'bapi', 'enhancement'],
            'functional': ['mm', 'sd', 'fi', 'co', 'hr', 'pp'],
            'security': ['authorization', 'role', 'profile', 'user'],
            'performance': ['performance', 'slow', 'timeout', 'memory']
        }
    
    def analyze_text_for_keywords(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for SAP keywords and classify incident type.
        
        Args:
            text: Text to analyze (description, short description, etc.)
            
        Returns:
            Dictionary with keyword analysis results
        """
        if not text or pd.isna(text):
            return {'keywords_found': [], 'classification': 'unknown', 'confidence': 0}
        
        # Clean and normalize text
        text_lower = str(text).lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        # Find matching keywords
        found_keywords = []
        for word in words:
            if word in self.sap_keywords:
                found_keywords.append(word)
        
        # Classify based on found keywords
        classification = self._classify_incident(found_keywords)
        confidence = len(found_keywords) / len(words) if words else 0
        
        return {
            'keywords_found': found_keywords,
            'classification': classification,
            'confidence': min(confidence * 10, 1.0),  # Scale confidence
            'total_words': len(words),
            'sap_word_count': len(found_keywords)
        }
    
    def analyze_keywords(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive keyword analysis for incident data.
        This method name matches the interface contract expected by main application.
        
        Args:
            df: DataFrame containing incident data
            
        Returns:
            Comprehensive keyword analysis results
        """
        try:
            if df is None or df.empty:
                return self._create_error_result("DataFrame is empty or None")
            
            self.logger.info(f"Starting keyword analysis for {len(df)} incidents")
            
            # Initialize keyword analysis results
            keyword_analysis = {
                'analysis_timestamp': datetime.now().isoformat(),
                'total_incidents_analyzed': len(df),
                'keyword_extraction': self._extract_keywords(df),
                'category_keywords': self._analyze_category_keywords(df),
                'urgency_patterns': self._analyze_urgency_keywords(df),
                'temporal_keywords': self._analyze_temporal_keywords(df),
                'sentiment_analysis': self._perform_sentiment_analysis(df),
                'keyword_trends': self._analyze_keyword_trends(df),
                'actionable_insights': self._generate_keyword_insights(df),
                'summary': {}
            }
            
            # Generate keyword summary
            keyword_analysis['summary'] = self._generate_keyword_summary(keyword_analysis)
            
            self.logger.info("Keyword analysis completed successfully")
            return keyword_analysis
            
        except Exception as e:
            error_msg = f"Error in keyword analysis: {str(e)}"
            self.logger.error(error_msg)
            return self._create_error_result(error_msg)

    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result following defensive programming."""
        return {
            'analysis_timestamp': datetime.now().isoformat(),
            'error': error_message,
            'total_incidents_analyzed': 0,
            'keyword_extraction': {},
            'category_keywords': {},
            'urgency_patterns': {},
            'temporal_keywords': {},
            'sentiment_analysis': {},
            'keyword_trends': {},
            'actionable_insights': [],
            'summary': {'status': 'failed', 'error': error_message}
        }

    def _extract_keywords(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract and analyze keywords from incident descriptions."""
        try:
            keyword_extraction = {
                'top_keywords': {},
                'keyword_frequency': {},
                'unique_keywords': 0,
                'total_keywords': 0
            }
            
            # Combine text fields for keyword extraction
            text_fields = ['short_description', 'description', 'work_notes']
            combined_text = []
            
            for field in text_fields:
                if field in df.columns:
                    text_data = df[field].dropna().astype(str)
                    combined_text.extend(text_data.tolist())
            
            if combined_text:
                # Simple keyword extraction (can be enhanced with NLP libraries)
                all_words = []
                for text in combined_text:
                    # Basic text processing
                    words = text.lower().split()
                    # Filter out common stop words and short words
                    filtered_words = [word.strip('.,!?;:"()[]{}') for word in words 
                                    if len(word) > 3 and word.lower() not in 
                                    ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'she', 'use', 'way', 'will', 'with']]
                    all_words.extend(filtered_words)
                
                # Calculate keyword frequencies
                from collections import Counter
                word_counts = Counter(all_words)
                
                keyword_extraction['top_keywords'] = dict(word_counts.most_common(20))
                keyword_extraction['keyword_frequency'] = dict(word_counts)
                keyword_extraction['unique_keywords'] = len(word_counts)
                keyword_extraction['total_keywords'] = sum(word_counts.values())
            
            return keyword_extraction
            
        except Exception as e:
            self.logger.error(f"Error extracting keywords: {str(e)}")
            return {'error': str(e)}

    def _analyze_category_keywords(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze keywords by incident category."""
        try:
            category_keywords = {}
            
            if 'category' in df.columns and 'short_description' in df.columns:
                categories = df['category'].dropna().unique()
                
                for category in categories[:10]:  # Limit to top 10 categories
                    category_data = df[df['category'] == category]['short_description'].dropna()
                    if not category_data.empty:
                        # Extract keywords for this category
                        category_text = ' '.join(category_data.astype(str))
                        words = category_text.lower().split()
                        
                        # Simple frequency analysis
                        from collections import Counter
                        word_counts = Counter([word.strip('.,!?;:"()[]{}') for word in words 
                                             if len(word) > 3])
                        
                        category_keywords[category] = {
                            'top_keywords': dict(word_counts.most_common(10)),
                            'total_incidents': len(category_data),
                            'keyword_diversity': len(word_counts)
                        }
            
            return category_keywords
            
        except Exception as e:
            self.logger.error(f"Error analyzing category keywords: {str(e)}")
            return {'error': str(e)}

    def _analyze_urgency_keywords(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze keyword patterns by urgency level."""
        try:
            urgency_patterns = {}
            
            if 'urgency' in df.columns and 'short_description' in df.columns:
                urgency_levels = df['urgency'].dropna().unique()
                
                for urgency in sorted(urgency_levels):
                    urgency_data = df[df['urgency'] == urgency]['short_description'].dropna()
                    if not urgency_data.empty:
                        # Extract keywords for this urgency level
                        urgency_text = ' '.join(urgency_data.astype(str))
                        words = urgency_text.lower().split()
                        
                        # Identify urgency-specific keywords
                        from collections import Counter
                        word_counts = Counter([word.strip('.,!?;:"()[]{}') for word in words 
                                             if len(word) > 3])
                        
                        urgency_patterns[f"urgency_{urgency}"] = {
                            'characteristic_keywords': dict(word_counts.most_common(8)),
                            'incident_count': len(urgency_data),
                            'urgency_indicators': self._identify_urgency_indicators(word_counts)
                        }
            
            return urgency_patterns
            
        except Exception as e:
            self.logger.error(f"Error analyzing urgency keywords: {str(e)}")
            return {'error': str(e)}

    def _identify_urgency_indicators(self, word_counts: 'Counter') -> List[str]:
        """Identify words that indicate urgency levels."""
        urgency_indicators = []
        high_urgency_words = ['critical', 'urgent', 'emergency', 'immediate', 'asap', 'priority', 'down', 'outage', 'failure']
        
        for word in high_urgency_words:
            if word in word_counts and word_counts[word] > 1:
                urgency_indicators.append(word)
        
        return urgency_indicators

    def _analyze_temporal_keywords(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze keyword patterns over time."""
        try:
            temporal_keywords = {}
            
            if 'sys_created_on' in df.columns and 'short_description' in df.columns:
                df_temporal = df.copy()
                df_temporal['sys_created_on'] = pd.to_datetime(df_temporal['sys_created_on'], errors='coerce')
                df_temporal = df_temporal.dropna(subset=['sys_created_on'])
                
                if not df_temporal.empty:
                    # Analyze keywords by month
                    df_temporal['month'] = df_temporal['sys_created_on'].dt.to_period('M')
                    monthly_keywords = {}
                    
                    for month in df_temporal['month'].unique()[-6:]:  # Last 6 months
                        month_data = df_temporal[df_temporal['month'] == month]['short_description'].dropna()
                        if not month_data.empty:
                            month_text = ' '.join(month_data.astype(str))
                            words = month_text.lower().split()
                            
                            from collections import Counter
                            word_counts = Counter([word.strip('.,!?;:"()[]{}') for word in words 
                                                 if len(word) > 3])
                            
                            monthly_keywords[str(month)] = {
                                'top_keywords': dict(word_counts.most_common(5)),
                                'incident_count': len(month_data)
                            }
                    
                    temporal_keywords['monthly_patterns'] = monthly_keywords
            
            return temporal_keywords
            
        except Exception as e:
            self.logger.error(f"Error analyzing temporal keywords: {str(e)}")
            return {'error': str(e)}

    def _perform_sentiment_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform basic sentiment analysis on incident descriptions."""
        try:
            sentiment_analysis = {
                'overall_sentiment': 'neutral',
                'sentiment_distribution': {},
                'negative_indicators': [],
                'positive_indicators': []
            }
            
            if 'short_description' in df.columns:
                descriptions = df['short_description'].dropna().astype(str)
                
                # Simple sentiment indicators
                negative_words = ['error', 'fail', 'problem', 'issue', 'broken', 'down', 'unable', 'cannot', 'not working']
                positive_words = ['resolved', 'fixed', 'working', 'completed', 'successful', 'restored']
                
                negative_count = 0
                positive_count = 0
                
                for desc in descriptions:
                    desc_lower = desc.lower()
                    if any(word in desc_lower for word in negative_words):
                        negative_count += 1
                    if any(word in desc_lower for word in positive_words):
                        positive_count += 1
                
                total_analyzed = len(descriptions)
                sentiment_analysis['sentiment_distribution'] = {
                    'negative_percentage': (negative_count / total_analyzed) * 100 if total_analyzed > 0 else 0,
                    'positive_percentage': (positive_count / total_analyzed) * 100 if total_analyzed > 0 else 0,
                    'neutral_percentage': ((total_analyzed - negative_count - positive_count) / total_analyzed) * 100 if total_analyzed > 0 else 0
                }
                
                # Determine overall sentiment
                if negative_count > positive_count * 1.5:
                    sentiment_analysis['overall_sentiment'] = 'negative'
                elif positive_count > negative_count * 1.5:
                    sentiment_analysis['overall_sentiment'] = 'positive'
                else:
                    sentiment_analysis['overall_sentiment'] = 'neutral'
            
            return sentiment_analysis
            
        except Exception as e:
            self.logger.error(f"Error performing sentiment analysis: {str(e)}")
            return {'error': str(e)}

    def _analyze_keyword_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trending keywords and patterns."""
        try:
            keyword_trends = {
                'emerging_keywords': [],
                'declining_keywords': [],
                'stable_keywords': [],
                'trend_analysis': 'completed'
            }
            
            # This is a simplified trend analysis
            # In a production system, you'd compare with historical data
            if 'short_description' in df.columns:
                # Extract recent vs older incidents for trend comparison
                if 'sys_created_on' in df.columns:
                    df_dates = df.copy()
                    df_dates['sys_created_on'] = pd.to_datetime(df_dates['sys_created_on'], errors='coerce')
                    df_dates = df_dates.dropna(subset=['sys_created_on'])
                    
                    if not df_dates.empty:
                        # Split into recent and older periods
                        median_date = df_dates['sys_created_on'].median()
                        recent_data = df_dates[df_dates['sys_created_on'] >= median_date]
                        older_data = df_dates[df_dates['sys_created_on'] < median_date]
                        
                        # Compare keyword frequencies
                        if not recent_data.empty and not older_data.empty:
                            recent_keywords = self._extract_simple_keywords(recent_data['short_description'])
                            older_keywords = self._extract_simple_keywords(older_data['short_description'])
                            
                            # Identify trends (simplified)
                            for keyword in recent_keywords:
                                if keyword in older_keywords:
                                    if recent_keywords[keyword] > older_keywords[keyword] * 1.5:
                                        keyword_trends['emerging_keywords'].append(keyword)
                                    elif recent_keywords[keyword] < older_keywords[keyword] * 0.5:
                                        keyword_trends['declining_keywords'].append(keyword)
                                    else:
                                        keyword_trends['stable_keywords'].append(keyword)
            
            return keyword_trends
            
        except Exception as e:
            self.logger.error(f"Error analyzing keyword trends: {str(e)}")
            return {'error': str(e)}

    def _extract_simple_keywords(self, text_series: pd.Series) -> Dict[str, int]:
        """Extract simple keyword frequencies from text series."""
        try:
            combined_text = ' '.join(text_series.dropna().astype(str))
            words = combined_text.lower().split()
            
            from collections import Counter
            word_counts = Counter([word.strip('.,!?;:"()[]{}') for word in words 
                                 if len(word) > 3])
            
            return dict(word_counts.most_common(20))
            
        except Exception as e:
            return {}

    def _generate_keyword_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate actionable insights from keyword analysis."""
        try:
            insights = []
            
            # Analyze common issues
            if 'short_description' in df.columns:
                descriptions = df['short_description'].dropna().astype(str)
                combined_text = ' '.join(descriptions).lower()
                
                # Identify common problem areas
                if 'network' in combined_text and combined_text.count('network') > 10:
                    insights.append("Network-related issues are frequently reported - consider network infrastructure review")
                
                if 'password' in combined_text and combined_text.count('password') > 15:
                    insights.append("Password-related incidents are common - consider implementing self-service password reset")
                
                if 'email' in combined_text and combined_text.count('email') > 20:
                    insights.append("Email issues are prevalent - review email system performance and user training")
                
                if 'access' in combined_text and combined_text.count('access') > 25:
                    insights.append("Access-related problems are frequent - review access management processes")
            
            if not insights:
                insights.append("Keyword analysis completed - no major patterns requiring immediate attention identified")
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating keyword insights: {str(e)}")
            return ["Unable to generate keyword insights due to analysis error"]

    def _generate_keyword_summary(self, keyword_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of keyword analysis results."""
        try:
            summary = {
                'analysis_status': 'completed',
                'total_incidents': keyword_analysis.get('total_incidents_analyzed', 0),
                'unique_keywords_found': keyword_analysis.get('keyword_extraction', {}).get('unique_keywords', 0),
                'top_keyword': 'Not identified',
                'sentiment_overview': keyword_analysis.get('sentiment_analysis', {}).get('overall_sentiment', 'neutral'),
                'key_insights_count': len(keyword_analysis.get('actionable_insights', [])),
                'analysis_quality': 'good'
            }
            
            # Identify top keyword
            top_keywords = keyword_analysis.get('keyword_extraction', {}).get('top_keywords', {})
            if top_keywords:
                summary['top_keyword'] = max(top_keywords, key=top_keywords.get)
            
            # Assess analysis quality
            if summary['unique_keywords_found'] > 100:
                summary['analysis_quality'] = 'excellent'
            elif summary['unique_keywords_found'] > 50:
                summary['analysis_quality'] = 'good'
            else:
                summary['analysis_quality'] = 'limited'
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating keyword summary: {str(e)}")
            return {'analysis_status': 'error', 'error': str(e)}
    
    def _classify_incident(self, keywords: List[str]) -> str:
        """Classify incident based on found keywords."""
        if not keywords:
            return 'non_sap'
        
        # Score each classification category
        category_scores = {}
        for category, category_keywords in self.classification_rules.items():
            score = sum(1 for kw in keywords if kw in category_keywords)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            return max(category_scores, key=category_scores.get)
        
        return 'sap_general'
    
    def learn_from_data(self, df: pd.DataFrame, text_columns: List[str] = None) -> Dict[str, Any]:
        """
        Learn new keywords from incident data.
        
        Args:
            df: DataFrame containing incident data
            text_columns: List of columns to analyze for keywords
            
        Returns:
            Dictionary with learning results
        """
        if text_columns is None:
            text_columns = ['Short Description', 'Description', 'Work notes']
        
        # Available columns in the dataframe
        available_columns = [col for col in text_columns if col in df.columns]
        
        if not available_columns:
            return {'error': 'No text columns found for learning'}
        
        # Extract all text
        all_text = []
        for col in available_columns:
            text_series = df[col].dropna().astype(str)
            all_text.extend(text_series.tolist())
        
        # Find potential new keywords
        new_keywords = self._extract_potential_keywords(all_text)
        
        # Score and filter keywords
        scored_keywords = self._score_keywords(new_keywords, all_text)
        
        # Add high-scoring keywords
        added_keywords = []
        for keyword, score in scored_keywords.items():
            if score > 0.1 and keyword not in self.sap_keywords:  # Threshold for adding
                self.sap_keywords.add(keyword)
                added_keywords.append(keyword)
        
        return {
            'new_keywords_added': added_keywords,
            'total_keywords': len(self.sap_keywords),
            'analyzed_texts': len(all_text),
            'potential_keywords_found': len(new_keywords)
        }
    
    def _extract_potential_keywords(self, texts: List[str]) -> Set[str]:
        """Extract potential SAP-related keywords from texts."""
        # Common SAP patterns
        sap_patterns = [
            r'\bt\w{3}\b',  # Transaction codes (T followed by 3 chars)
            r'\b[a-z]{2}\d{2}\b',  # Module codes (2 letters + 2 digits)
            r'\bse\d{2}\b',  # SE transactions
            r'\bsm\d{2}\b',  # SM transactions
            r'\bst\d{2}\b',  # ST transactions
            r'\bsu\d{2}\b',  # SU transactions
        ]
        
        potential_keywords = set()
        
        for text in texts:
            if not text:
                continue
                
            text_lower = str(text).lower()
            
            # Extract pattern-based keywords
            for pattern in sap_patterns:
                matches = re.findall(pattern, text_lower)
                potential_keywords.update(matches)
            
            # Extract words that appear with known SAP keywords
            words = re.findall(r'\b\w+\b', text_lower)
            for i, word in enumerate(words):
                if word in self.sap_keywords:
                    # Add surrounding words as potential keywords
                    if i > 0:
                        potential_keywords.add(words[i-1])
                    if i < len(words) - 1:
                        potential_keywords.add(words[i+1])
        
        return potential_keywords
    
    def _score_keywords(self, keywords: Set[str], texts: List[str]) -> Dict[str, float]:
        """Score potential keywords based on frequency and context."""
        keyword_scores = {}
        total_texts = len(texts)
        
        for keyword in keywords:
            # Count occurrences
            occurrences = sum(1 for text in texts if keyword in str(text).lower())
            
            # Calculate frequency score
            frequency_score = occurrences / total_texts
            
            # Bonus for appearing with existing SAP keywords
            context_score = 0
            for text in texts:
                if keyword in str(text).lower():
                    text_words = set(re.findall(r'\b\w+\b', str(text).lower()))
                    sap_word_overlap = len(text_words.intersection(self.sap_keywords))
                    if sap_word_overlap > 0:
                        context_score += sap_word_overlap / len(text_words)
            
            context_score = context_score / total_texts if total_texts > 0 else 0
            
            # Combined score
            keyword_scores[keyword] = frequency_score + (context_score * 0.5)
        
        return keyword_scores
    
    def get_keyword_statistics(self) -> Dict[str, Any]:
        """Get statistics about current keywords."""
        return {
            'total_keywords': len(self.sap_keywords),
            'classification_categories': list(self.classification_rules.keys()),
            'keywords_by_category': {
                cat: [kw for kw in keywords if kw in self.sap_keywords]
                for cat, keywords in self.classification_rules.items()
            }
        }
    
    def save_keywords(self, file_path: str):
        """Save current keywords to file."""
        try:
            data = {
                'keywords': list(self.sap_keywords),
                'rules': self.classification_rules,
                'metadata': {
                    'total_keywords': len(self.sap_keywords),
                    'last_updated': pd.Timestamp.now().isoformat()
                }
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving keywords: {e}")
            raise


# Alias for backward compatibility
KeywordManager = IncidentKeywordManager