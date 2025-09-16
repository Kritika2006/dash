# cfo_dashboard/backend/services/llm_service.py
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from ..core.schemas import AskPayload
import re
import json
from typing import Dict, List, Any

class EnhancedLLMAdvisory:
    def __init__(self, model_name: str = "google/flan-t5-base"):
        print("⚡ Initializing Enhanced FLAN-T5-base model...")
        self.available = False
        try:
            device = 0 if torch.cuda.is_available() else -1
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.text2text = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)
            self.available = True
            print("✅ Enhanced model loaded successfully.")
        except Exception as e:
            print(f"⚠ Failed to load model: {e}")
            self.text2text = None
        
        # Financial domain knowledge base
        self.financial_knowledge = self._load_financial_knowledge()
    
    def _load_financial_knowledge(self) -> Dict[str, Any]:
        """Load financial domain knowledge and best practices."""
        return {
            "kpi_benchmarks": {
                "profit_margin": {"excellent": 0.20, "good": 0.10, "poor": 0.05},
                "debt_ratio": {"excellent": 0.30, "good": 0.50, "poor": 0.70},
                "runway_months": {"excellent": 18, "good": 12, "poor": 6},
                "burn_rate": {"excellent": 0.05, "good": 0.10, "poor": 0.20}
            },
            "industry_insights": {
                "saas": {"typical_margin": 0.15, "growth_rate": 0.30, "churn_rate": 0.05},
                "ecommerce": {"typical_margin": 0.10, "growth_rate": 0.20, "seasonality": True},
                "manufacturing": {"typical_margin": 0.08, "growth_rate": 0.10, "capital_intensive": True},
                "services": {"typical_margin": 0.12, "growth_rate": 0.15, "labor_intensive": True}
            },
            "risk_factors": {
                "high_debt": "High debt levels increase financial risk and reduce flexibility",
                "low_cash": "Insufficient cash reserves limit growth opportunities and increase vulnerability",
                "negative_margin": "Negative profit margins indicate unsustainable business model",
                "short_runway": "Short runway requires immediate action to secure funding or reduce costs"
            },
            "recommendations": {
                "cost_optimization": [
                    "Review and optimize operational expenses",
                    "Implement automation to reduce labor costs",
                    "Negotiate better terms with suppliers",
                    "Consider outsourcing non-core functions"
                ],
                "revenue_growth": [
                    "Focus on customer acquisition and retention",
                    "Develop new products or services",
                    "Expand into new markets",
                    "Improve pricing strategy"
                ],
                "cash_management": [
                    "Implement stricter payment terms",
                    "Optimize inventory levels",
                    "Consider invoice factoring",
                    "Establish credit lines"
                ],
                "risk_mitigation": [
                    "Diversify revenue streams",
                    "Build emergency cash reserves",
                    "Implement robust financial controls",
                    "Regular financial health monitoring"
                ]
            }
        }
    
    def _analyze_financial_health(self, finance: Any) -> Dict[str, Any]:
        """Analyze financial health based on KPIs."""
        analysis = {
            "overall_health": "unknown",
            "strengths": [],
            "weaknesses": [],
            "recommendations": [],
            "risk_level": "medium"
        }
        
        # Calculate key metrics
        profit_margin = (finance.revenue - finance.expenses) / finance.revenue if finance.revenue > 0 else 0
        debt_ratio = finance.liabilities / finance.revenue if finance.revenue > 0 else 0
        runway_months = finance.cash / (finance.expenses / 12) if finance.expenses > 0 else 0
        
        # Analyze profit margin
        if profit_margin >= 0.20:
            analysis["strengths"].append("Excellent profit margin")
        elif profit_margin >= 0.10:
            analysis["strengths"].append("Good profit margin")
        elif profit_margin >= 0.05:
            analysis["weaknesses"].append("Low profit margin")
        else:
            analysis["weaknesses"].append("Negative profit margin - unsustainable")
            analysis["risk_level"] = "high"
        
        # Analyze debt ratio
        if debt_ratio <= 0.30:
            analysis["strengths"].append("Low debt levels")
        elif debt_ratio <= 0.50:
            analysis["strengths"].append("Manageable debt levels")
        elif debt_ratio <= 0.70:
            analysis["weaknesses"].append("High debt levels")
        else:
            analysis["weaknesses"].append("Very high debt levels - high risk")
            analysis["risk_level"] = "high"
        
        # Analyze runway
        if runway_months >= 18:
            analysis["strengths"].append("Strong cash position")
        elif runway_months >= 12:
            analysis["strengths"].append("Adequate cash runway")
        elif runway_months >= 6:
            analysis["weaknesses"].append("Short cash runway")
        else:
            analysis["weaknesses"].append("Critical cash shortage")
            analysis["risk_level"] = "high"
        
        # Generate recommendations
        if profit_margin < 0.10:
            analysis["recommendations"].extend(self.financial_knowledge["recommendations"]["cost_optimization"])
        
        if runway_months < 12:
            analysis["recommendations"].extend(self.financial_knowledge["recommendations"]["cash_management"])
        
        if debt_ratio > 0.50:
            analysis["recommendations"].extend(self.financial_knowledge["recommendations"]["risk_mitigation"])
        
        # Determine overall health
        if len(analysis["strengths"]) > len(analysis["weaknesses"]):
            analysis["overall_health"] = "good"
        elif len(analysis["weaknesses"]) > len(analysis["strengths"]):
            analysis["overall_health"] = "poor"
        else:
            analysis["overall_health"] = "fair"
        
        return analysis
    
    def _generate_contextual_prompt(self, payload: AskPayload, analysis: Dict[str, Any]) -> str:
        """Generate a contextual prompt with financial analysis."""
        finance = payload.finance
        
        # Basic financial context
        context = f"""
Financial Context:
- Revenue: ${finance.revenue:,.0f}
- Expenses: ${finance.expenses:,.0f}
- Profit: ${finance.revenue - finance.expenses:,.0f}
- Profit Margin: {((finance.revenue - finance.expenses) / finance.revenue * 100):.1f}%
- Liabilities: ${finance.liabilities:,.0f}
- Debt Ratio: {(finance.liabilities / finance.revenue * 100):.1f}%
- Cash: ${finance.cash if finance.cash else 0:,.0f}
- Burn Rate: ${finance.burn_rate:,.0f}/month
- Runway: {(finance.cash / (finance.burn_rate / 12)) if finance.burn_rate > 0 else 0:.1f} months

Financial Health Analysis:
- Overall Health: {analysis['overall_health'].title()}
- Risk Level: {analysis['risk_level'].title()}
- Strengths: {', '.join(analysis['strengths']) if analysis['strengths'] else 'None identified'}
- Weaknesses: {', '.join(analysis['weaknesses']) if analysis['weaknesses'] else 'None identified'}

Key Recommendations:
{chr(10).join(f"- {rec}" for rec in analysis['recommendations'][:3]) if analysis['recommendations'] else '- No specific recommendations'}

Question: {payload.question}

Please provide a comprehensive financial advisory response that:
1. Directly addresses the question
2. Considers the current financial position
3. Provides actionable recommendations
4. Explains the reasoning behind suggestions
5. Mentions relevant financial metrics and benchmarks
"""
        return context
    
    def ask(self, payload: AskPayload) -> str:
        """Enhanced ask method with financial domain knowledge."""
        finance = payload.finance
        
        # Perform financial health analysis
        analysis = self._analyze_financial_health(finance)
        
        # Generate contextual prompt
        prompt = self._generate_contextual_prompt(payload, analysis)
        
        try:
            if self.available and self.text2text is not None:
                response = self.text2text(prompt, max_new_tokens=300, do_sample=True, temperature=0.7)
                ai_response = response[0]["generated_text"]
                
                # Enhance response with structured insights
                enhanced_response = self._enhance_response(ai_response, analysis, finance)
                return enhanced_response
            else:
                # Fallback to rule-based response
                return self._generate_fallback_response(payload, analysis)
        except Exception as e:
            return f"⚠ Error during model inference: {e}. {self._generate_fallback_response(payload, analysis)}"
    
    def _enhance_response(self, ai_response: str, analysis: Dict[str, Any], finance: Any) -> str:
        """Enhance AI response with structured insights."""
        enhanced = f"{ai_response}\n\n"
        
        # Add financial health summary
        enhanced += f"📊 **Financial Health Summary:**\n"
        enhanced += f"- Overall Health: {analysis['overall_health'].title()}\n"
        enhanced += f"- Risk Level: {analysis['risk_level'].title()}\n\n"
        
        # Add key metrics
        profit_margin = (finance.revenue - finance.expenses) / finance.revenue * 100 if finance.revenue > 0 else 0
        enhanced += f"📈 **Key Metrics:**\n"
        enhanced += f"- Profit Margin: {profit_margin:.1f}%\n"
        enhanced += f"- Debt Ratio: {(finance.liabilities / finance.revenue * 100):.1f}%\n"
        enhanced += f"- Cash Runway: {(finance.cash / (finance.burn_rate / 12)) if finance.burn_rate > 0 else 0:.1f} months\n\n"
        
        # Add recommendations
        if analysis['recommendations']:
            enhanced += f"💡 **Priority Recommendations:**\n"
            for i, rec in enumerate(analysis['recommendations'][:3], 1):
                enhanced += f"{i}. {rec}\n"
        
        return enhanced
    
    def _generate_fallback_response(self, payload: AskPayload, analysis: Dict[str, Any]) -> str:
        """Generate fallback response when AI model is unavailable."""
        finance = payload.finance
        question = payload.question.lower()
        
        response = "Based on your financial data, here's my analysis:\n\n"
        
        # Analyze the question type
        if "profit" in question or "margin" in question:
            profit_margin = (finance.revenue - finance.expenses) / finance.revenue * 100 if finance.revenue > 0 else 0
            if profit_margin > 20:
                response += f"✅ Your profit margin of {profit_margin:.1f}% is excellent and above industry standards.\n"
            elif profit_margin > 10:
                response += f"📊 Your profit margin of {profit_margin:.1f}% is good but could be improved.\n"
            else:
                response += f"⚠️ Your profit margin of {profit_margin:.1f}% is concerning and needs immediate attention.\n"
        
        if "cash" in question or "runway" in question:
            runway = (finance.cash / (finance.burn_rate / 12)) if finance.burn_rate > 0 else 0
            if runway > 18:
                response += f"✅ You have a strong cash position with {runway:.1f} months runway.\n"
            elif runway > 12:
                response += f"📊 Your cash runway of {runway:.1f} months is adequate.\n"
            else:
                response += f"🚨 Critical: You only have {runway:.1f} months of runway left. Immediate action required.\n"
        
        if "debt" in question or "liability" in question:
            debt_ratio = finance.liabilities / finance.revenue if finance.revenue > 0 else 0
            if debt_ratio < 0.3:
                response += f"✅ Your debt ratio of {debt_ratio:.2f} is excellent.\n"
            elif debt_ratio < 0.5:
                response += f"📊 Your debt ratio of {debt_ratio:.2f} is manageable.\n"
            else:
                response += f"⚠️ Your debt ratio of {debt_ratio:.2f} is high and poses financial risk.\n"
        
        # Add general recommendations        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        if analysis['recommendations']:
            response += f"\n💡 **Key Recommendations:**\n"
            for i, rec in enumerate(analysis['recommendations'][:3], 1):
                response += f"{i}. {rec}\n"
        
        return response

# Create a single instance to be shared across the app
llm_instance = EnhancedLLMAdvisory()