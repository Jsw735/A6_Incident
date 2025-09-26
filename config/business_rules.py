#!/usr/bin/env python3
"""
Business Rules for Incident Reporting System.
Defines health scoring weights, SLA thresholds, and executive-level business logic.
Follows PEP 8 standards and implements comprehensive error handling.
"""

from typing import Dict, Any, List, Tuple
import logging
from datetime import datetime, timedelta


class BusinessRules:
    """
    Centralized business rules for incident management system.
    
    This class implements CIO-level health scoring and business logic
    following the requirements for weighted health calculations and
    October backlog clearance targets.
    """
    
    def __init__(self):
        """Initialize business rules with executive requirements."""
        self.logger = logging.getLogger(__name__)
        
        # Health score weights as specified in requirements
        # Backlog weighted most (40%), closure next (30%), accuracy (20%), training (10%)
        self.health_score_weights = {
            'backlog': 0.4,      # Highest weight - backlog reduction priority
            'closure': 0.3,      # Second priority - closure rate performance
            'accuracy': 0.2,     # Third priority - ticket accuracy
            'training': 0.1      # Fourth priority - training percentage
        }
        
        # SLA thresholds based on priority levels from requirements table
        self.sla_thresholds = {
            'P0': {'hours': 4, 'urgency': 'High', 'description': 'Critical Incident'},
            'P1': {'hours': 24, 'urgency': 'Medium', 'description': 'High Incident'},
            'P2': {'hours': 72, 'urgency': 'Medium', 'description': 'Medium Incident'},
            'P3': {'hours': 168, 'urgency': 'Low', 'description': 'Low Incident'}
        }
        
        # Health status thresholds for executive reporting
        self.health_thresholds = {
            'excellent': 85.0,
            'good': 75.0,
            'fair': 65.0,
            'needs_attention': 0.0
        }
        
        # Target metrics for October backlog clearance
        self.target_metrics = {
            'max_backlog': 300,              # Target maximum open incidents
            'target_closure_rate': 85.0,     # Target closure rate percentage
            'min_accuracy': 80.0,            # Minimum ticket accuracy
            'optimal_training_range': (10.0, 15.0)  # Optimal training percentage range
        }
        
        # Urgency mapping for display purposes
        self.urgency_mapping = {
            '1': 'High',
            '2': 'Medium',
            '3': 'Medium',
            '4': 'Low'
        }
    
    def calculate_executive_health_score(self, core_metrics: Dict, trend_metrics: Dict,
                                       quality_metrics: Dict, training_metrics: Dict) -> Dict[str, Any]:
        """
        Calculate weighted health score for executive dashboard.
        
        Implements the specified weighting system:
        - Backlog: 40% (highest priority)
        - Closure Rate: 30% 
        - Accuracy: 20%
        - Training: 10%
        
        Args:
            core_metrics: Basic incident counts and rates
            trend_metrics: Trend analysis results
            quality_metrics: Ticket accuracy metrics
            training_metrics: Training classification metrics
            
        Returns:
            Complete health score analysis with recommendations
        """
        try:
            # Calculate individual component scores
            backlog_score = self._calculate_backlog_score(
                core_metrics.get('open_incidents', 0),
                trend_metrics.get('backlog_trend', 'stable')
            )
            
            closure_score = self._calculate_closure_score(
                core_metrics.get('closure_rate_percentage', 0),
                trend_metrics.get('closure_rate_trend', 'stable')
            )
            
            accuracy_score = self._calculate_accuracy_score(
                quality_metrics.get('accuracy_percentage', 75.0)
            )
            
            training_score = self._calculate_training_score(
                training_metrics.get('training_percentage', 15.0)
            )
            
            # Calculate weighted overall health score
            overall_health = (
                backlog_score * self.health_score_weights['backlog'] +
                closure_score * self.health_score_weights['closure'] +
                accuracy_score * self.health_score_weights['accuracy'] +
                training_score * self.health_score_weights['training']
            )
            
            # Determine health status and color
            health_status, health_color = self._determine_health_status(overall_health)
            
            # Generate actionable recommendations
            recommendations = self._generate_health_recommendations(
                backlog_score, closure_score, accuracy_score, training_score
            )
            
            # Calculate trend direction
            health_trend = self._calculate_health_trend(
                overall_health, trend_metrics
            )
            
            return {
                'overall_health_score': round(overall_health, 1),
                'health_status': health_status,
                'health_color': health_color,
                'component_scores': {
                    'backlog_score': round(backlog_score, 1),
                    'closure_score': round(closure_score, 1),
                    'accuracy_score': round(accuracy_score, 1),
                    'training_score': round(training_score, 1)
                },
                'weights_applied': self.health_score_weights,
                'recommendations': recommendations,
                'health_trend': health_trend,
                'target_metrics': self.target_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Health score calculation error: {str(e)}")
            return self._get_default_health_score()
    
    def _calculate_backlog_score(self, open_incidents: int, trend: str) -> float:
        """
        Calculate backlog component score (40% weight).
        
        This is the highest weighted component as backlog reduction
        is the primary focus for October targets.
        """
        try:
            base_score = 100.0
            
            # Score based on absolute backlog size
            if open_incidents <= self.target_metrics['max_backlog']:
                base_score = 100.0  # Excellent - at or below target
            elif open_incidents <= 500:
                base_score = 85.0   # Good - manageable backlog
            elif open_incidents <= 750:
                base_score = 70.0   # Fair - elevated backlog
            elif open_incidents <= 1000:
                base_score = 55.0   # Poor - high backlog
            else:
                # Severe penalty for very large backlogs
                base_score = max(30.0, 100.0 - (open_incidents - 1000) * 0.05)
            
            # Adjust based on trend direction
            if trend == 'improving':
                base_score = min(100.0, base_score * 1.1)  # 10% bonus for improvement
            elif trend == 'declining':
                base_score = max(20.0, base_score * 0.9)   # 10% penalty for decline
            
            return base_score
            
        except Exception as e:
            self.logger.error(f"Backlog score calculation error: {str(e)}")
            return 70.0
    
    def _calculate_closure_score(self, closure_rate: float, trend: str) -> float:
        """
        Calculate closure rate component score (30% weight).
        
        Second highest priority as closure rate directly impacts
        backlog reduction goals.
        """
        try:
            # Base score from closure rate percentage
            if closure_rate >= self.target_metrics['target_closure_rate']:
                base_score = 100.0  # Meeting or exceeding target
            elif closure_rate >= 75.0:
                base_score = 85.0   # Good performance
            elif closure_rate >= 65.0:
                base_score = 70.0   # Acceptable performance
            elif closure_rate >= 50.0:
                base_score = 55.0   # Below expectations
            else:
                base_score = max(30.0, closure_rate * 0.8)  # Poor performance
            
            # Trend adjustment
            if trend == 'improving':
                base_score = min(100.0, base_score * 1.05)  # 5% bonus for improvement
            elif trend == 'declining':
                base_score = max(25.0, base_score * 0.95)   # 5% penalty for decline
            
            return base_score
            
        except Exception as e:
            self.logger.error(f"Closure score calculation error: {str(e)}")
            return 75.0
    
    def _calculate_accuracy_score(self, accuracy_percentage: float) -> float:
        """
        Calculate ticket accuracy component score (20% weight).
        
        Important for operational efficiency and proper resource allocation.
        """
        try:
            if accuracy_percentage >= 95.0:
                return 100.0  # Excellent accuracy
            elif accuracy_percentage >= self.target_metrics['min_accuracy']:
                return 90.0   # Meeting target
            elif accuracy_percentage >= 75.0:
                return 80.0   # Acceptable accuracy
            elif accuracy_percentage >= 65.0:
                return 70.0   # Below target
            else:
                return max(40.0, accuracy_percentage * 0.8)  # Poor accuracy
                
        except Exception as e:
            self.logger.error(f"Accuracy score calculation error: {str(e)}")
            return 75.0
    
    def _calculate_training_score(self, training_percentage: float) -> float:
        """
        Calculate training component score (10% weight).
        
        Lowest weight but important for identifying process improvement needs.
        """
        try:
            optimal_min, optimal_max = self.target_metrics['optimal_training_range']
            
            if optimal_min <= training_percentage <= optimal_max:
                return 100.0  # Optimal range
            elif training_percentage < optimal_min:
                # Too few training requests might indicate issues
                return max(70.0, training_percentage * 8.0)
            else:
                # Too many training requests might indicate process issues
                excess = training_percentage - optimal_max
                return max(60.0, 100.0 - (excess * 2.0))
                
        except Exception as e:
            self.logger.error(f"Training score calculation error: {str(e)}")
            return 85.0
    
    def _determine_health_status(self, health_score: float) -> Tuple[str, str]:
        """Determine health status and color based on score."""
        if health_score >= self.health_thresholds['excellent']:
            return 'Excellent', 'excellent'
        elif health_score >= self.health_thresholds['good']:
            return 'Good', 'good'
        elif health_score >= self.health_thresholds['fair']:
            return 'Fair', 'warning'
        else:
            return 'Needs Attention', 'danger'
    
    def _generate_health_recommendations(self, backlog_score: float, closure_score: float,
                                       accuracy_score: float, training_score: float) -> List[str]:
        """
        Generate actionable recommendations based on component scores.
        
        Provides specific, actionable guidance for improving health metrics.
        """
        recommendations = []
        
        # Backlog recommendations (highest priority)
        if backlog_score < 70:
            recommendations.append("URGENT: Implement aggressive backlog reduction strategy")
            recommendations.append("Consider additional resources for incident resolution")
            recommendations.append("Review and expedite high-priority incident processing")
        
        # Closure rate recommendations
        if closure_score < 70:
            recommendations.append("Focus on improving incident closure processes")
            recommendations.append("Review SLA compliance and resolution procedures")
            recommendations.append("Analyze closure bottlenecks and process inefficiencies")
        
        # Accuracy recommendations
        if accuracy_score < 80:
            recommendations.append("Enhance ticket classification training")
            recommendations.append("Review and update categorization guidelines")
            recommendations.append("Implement quality assurance checks for ticket routing")
        
        # Training recommendations
        if training_score < 70:
            recommendations.append("Optimize training request identification and handling")
            recommendations.append("Review training delivery processes and effectiveness")
        
        # Positive reinforcement when performing well
        if not recommendations:
            recommendations.append("Maintain current operational excellence")
            recommendations.append("Continue monitoring key performance indicators")
            recommendations.append("Consider sharing best practices across teams")
        
        return recommendations
    
    def _calculate_health_trend(self, current_health: float, trend_metrics: Dict) -> str:
        """
        Calculate health trend indicator based on component trends.
        
        This would typically compare with historical data.
        For now, uses component trend analysis.
        """
        try:
            # Analyze component trends to determine overall direction
            improving_trends = 0
            declining_trends = 0
            
            for trend_key in ['backlog_trend', 'closure_rate_trend', 'accuracy_trend']:
                trend_value = trend_metrics.get(trend_key, 'stable')
                if trend_value == 'improving':
                    improving_trends += 1
                elif trend_value == 'declining':
                    declining_trends += 1
            
            if improving_trends > declining_trends:
                return 'improving'
            elif declining_trends > improving_trends:
                return 'declining'
            else:
                return 'stable'
                
        except Exception:
            return 'stable'
    
    def _get_default_health_score(self) -> Dict[str, Any]:
        """Return default health score in case of calculation errors."""
        return {
            'overall_health_score': 75.0,
            'health_status': 'Fair',
            'health_color': 'warning',
            'component_scores': {
                'backlog_score': 75.0,
                'closure_score': 75.0,
                'accuracy_score': 75.0,
                'training_score': 75.0
            },
            'recommendations': ['Review system performance and data quality'],
            'health_trend': 'stable'
        }
    
    def get_health_score_weights(self) -> Dict[str, float]:
        """Get health score weights for external use."""
        return self.health_score_weights.copy()
    
    def get_sla_thresholds(self) -> Dict[str, Dict[str, Any]]:
        """Get SLA thresholds for external use."""
        return self.sla_thresholds.copy()
    
    def get_target_metrics(self) -> Dict[str, Any]:
        """Get target metrics for external use."""
        return self.target_metrics.copy()
    
    def get_urgency_mapping(self) -> Dict[str, str]:
        """Get urgency mapping for external use."""
        return self.urgency_mapping.copy()
    
    def validate_health_inputs(self, core_metrics: Dict, trend_metrics: Dict,
                             quality_metrics: Dict, training_metrics: Dict) -> bool:
        """
        Validate input metrics for health score calculation.
        
        Ensures all required metrics are present and within expected ranges.
        """
        try:
            required_core = ['open_incidents', 'closure_rate_percentage']
            required_quality = ['accuracy_percentage']
            required_training = ['training_percentage']
            
            # Check required fields
            for field in required_core:
                if field not in core_metrics:
                    self.logger.warning(f"Missing required core metric: {field}")
                    return False
            
            for field in required_quality:
                if field not in quality_metrics:
                    self.logger.warning(f"Missing required quality metric: {field}")
                    return False
            
            for field in required_training:
                if field not in training_metrics:
                    self.logger.warning(f"Missing required training metric: {field}")
                    return False
            
            # Validate ranges
            if not (0 <= quality_metrics.get('accuracy_percentage', 0) <= 100):
                self.logger.warning("Accuracy percentage out of valid range (0-100)")
                return False
            
            if not (0 <= training_metrics.get('training_percentage', 0) <= 100):
                self.logger.warning("Training percentage out of valid range (0-100)")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Health input validation error: {str(e)}")
            return False