# cfo_dashboard/backend/services/data_validation_service.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DataValidator:
    """Comprehensive data validation service for financial data."""
    
    def __init__(self):
        self.validation_rules = self._load_validation_rules()
        self.error_messages = []
        self.warning_messages = []
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules for different data types."""
        return {
            "financial_columns": {
                "revenue": {"min": 0, "max": 1e12, "required": True},
                "expenses": {"min": 0, "max": 1e12, "required": True},
                "profit": {"min": -1e12, "max": 1e12, "required": False},
                "cash": {"min": 0, "max": 1e12, "required": False},
                "liabilities": {"min": 0, "max": 1e12, "required": False},
                "assets": {"min": 0, "max": 1e12, "required": False}
            },
            "date_formats": [
                "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y-%m", "%m-%Y", "%Y"
            ],
            "text_patterns": {
                "business_name": r"^[a-zA-Z0-9\s\-\.&]+$",
                "category": r"^[a-zA-Z0-9\s\-]+$"
            }
        }
    
    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive dataframe validation."""
        self.error_messages = []
        self.warning_messages = []
        
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "data_quality_score": 0.0,
            "suggestions": []
        }
        
        if df.empty:
            validation_result["is_valid"] = False
            validation_result["errors"].append("DataFrame is empty")
            return validation_result
        
        # Basic structure validation
        self._validate_structure(df)
        
        # Data type validation
        self._validate_data_types(df)
        
        # Range validation
        self._validate_ranges(df)
        
        # Completeness validation
        self._validate_completeness(df)
        
        # Consistency validation
        self._validate_consistency(df)
        
        # Calculate data quality score
        quality_score = self._calculate_quality_score(df)
        
        validation_result.update({
            "is_valid": len(self.error_messages) == 0,
            "errors": self.error_messages,
            "warnings": self.warning_messages,
            "data_quality_score": quality_score,
            "suggestions": self._generate_suggestions(df)
        })
        
        return validation_result
    
    def _validate_structure(self, df: pd.DataFrame):
        """Validate basic dataframe structure."""
        # Check minimum rows
        if len(df) < 1:
            self.error_messages.append("DataFrame must have at least 1 row")
        
        # Check minimum columns
        if len(df.columns) < 2:
            self.error_messages.append("DataFrame must have at least 2 columns")
        
        # Check for duplicate columns
        if len(df.columns) != len(set(df.columns)):
            self.error_messages.append("DataFrame has duplicate column names")
        
        # Check for empty column names
        empty_columns = [col for col in df.columns if not str(col).strip()]
        if empty_columns:
            self.error_messages.append(f"DataFrame has empty column names: {empty_columns}")
    
    def _validate_data_types(self, df: pd.DataFrame):
        """Validate data types and format consistency."""
        for col in df.columns:
            col_data = df[col]
            
            # Check for mixed data types
            if col_data.dtype == 'object':
                # Try to identify the intended data type
                numeric_count = 0
                date_count = 0
                text_count = 0
                
                for value in col_data.dropna():
                    if self._is_numeric(value):
                        numeric_count += 1
                    elif self._is_date(value):
                        date_count += 1
                    else:
                        text_count += 1
                
                total_non_null = numeric_count + date_count + text_count
                if total_non_null > 0:
                    numeric_ratio = numeric_count / total_non_null
                    date_ratio = date_count / total_non_null
                    
                    if numeric_ratio > 0.8:
                        self.warning_messages.append(
                            f"Column '{col}' appears to be numeric but stored as text"
                        )
                    elif date_ratio > 0.8:
                        self.warning_messages.append(
                            f"Column '{col}' appears to be dates but stored as text"
                        )
    
    def _validate_ranges(self, df: pd.DataFrame):
        """Validate data ranges and outliers."""
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                col_data = df[col].dropna()
                
                if len(col_data) > 0:
                    # Check for negative values in financial columns
                    if any(keyword in col.lower() for keyword in ['revenue', 'cash', 'assets']):
                        negative_count = (col_data < 0).sum()
                        if negative_count > 0:
                            self.warning_messages.append(
                                f"Column '{col}' has {negative_count} negative values"
                            )
                    
                    # Check for extremely large values
                    max_value = col_data.max()
                    if max_value > 1e10:
                        self.warning_messages.append(
                            f"Column '{col}' has extremely large values (max: {max_value:,.0f})"
                        )
                    
                    # Check for outliers using IQR method
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                    if len(outliers) > 0:
                        outlier_ratio = len(outliers) / len(col_data)
                        if outlier_ratio > 0.1:  # More than 10% outliers
                            self.warning_messages.append(
                                f"Column '{col}' has {len(outliers)} potential outliers"
                            )
    
    def _validate_completeness(self, df: pd.DataFrame):
        """Validate data completeness."""
        total_cells = df.size
        null_cells = df.isnull().sum().sum()
        completeness_ratio = (total_cells - null_cells) / total_cells
        
        if completeness_ratio < 0.8:
            self.warning_messages.append(
                f"Low data completeness: {completeness_ratio:.1%} of cells have data"
            )
        
        # Check for completely empty columns
        empty_columns = df.columns[df.isnull().all()].tolist()
        if empty_columns:
            self.error_messages.append(f"Completely empty columns: {empty_columns}")
        
        # Check for completely empty rows
        empty_rows = df.isnull().all(axis=1).sum()
        if empty_rows > 0:
            self.warning_messages.append(f"{empty_rows} completely empty rows found")
    
    def _validate_consistency(self, df: pd.DataFrame):
        """Validate data consistency and relationships."""
        # Check for duplicate rows
        duplicate_rows = df.duplicated().sum()
        if duplicate_rows > 0:
            self.warning_messages.append(f"{duplicate_rows} duplicate rows found")
        
        # Check financial relationships if applicable
        if 'revenue' in df.columns and 'expenses' in df.columns:
            revenue_col = df['revenue']
            expenses_col = df['expenses']
            
            # Check for negative profit scenarios
            profit = revenue_col - expenses_col
            negative_profit_count = (profit < 0).sum()
            if negative_profit_count > 0:
                self.warning_messages.append(
                    f"{negative_profit_count} records show expenses exceeding revenue"
                )
        
        # Check for logical inconsistencies
        if 'cash' in df.columns and 'expenses' in df.columns:
            cash_col = df['cash']
            expenses_col = df['expenses']
            
            # Check if cash is reasonable compared to expenses
            monthly_expenses = expenses_col / 12
            unreasonable_cash = (cash_col > monthly_expenses * 24).sum()  # More than 2 years of expenses
            if unreasonable_cash > 0:
                self.warning_messages.append(
                    f"{unreasonable_cash} records show unusually high cash relative to expenses"
                )
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate overall data quality score (0-1)."""
        score = 1.0
        
        # Deduct for errors
        score -= len(self.error_messages) * 0.2
        
        # Deduct for warnings
        score -= len(self.warning_messages) * 0.05
        
        # Deduct for missing data
        completeness = (df.size - df.isnull().sum().sum()) / df.size
        score -= (1 - completeness) * 0.3
        
        # Deduct for data type issues
        object_columns = df.select_dtypes(include=['object']).columns
        if len(object_columns) > 0:
            type_issues = 0
            for col in object_columns:
                if self._is_numeric_column(df[col]):
                    type_issues += 1
            score -= (type_issues / len(object_columns)) * 0.2
        
        return max(0.0, min(1.0, score))
    
    def _generate_suggestions(self, df: pd.DataFrame) -> List[str]:
        """Generate data improvement suggestions."""
        suggestions = []
        
        # Data type suggestions
        for col in df.columns:
            if df[col].dtype == 'object':
                if self._is_numeric_column(df[col]):
                    suggestions.append(f"Consider converting column '{col}' to numeric type")
                elif self._is_date_column(df[col]):
                    suggestions.append(f"Consider converting column '{col}' to datetime type")
        
        # Missing data suggestions
        missing_data = df.isnull().sum()
        high_missing_cols = missing_data[missing_data > len(df) * 0.1].index.tolist()
        if high_missing_cols:
            suggestions.append(f"Consider imputing missing values in: {', '.join(high_missing_cols)}")
        
        # Outlier suggestions
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].dtype in ['int64', 'float64']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[col][(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
                if len(outliers) > len(df) * 0.05:
                    suggestions.append(f"Review outliers in column '{col}'")
        
        return suggestions
    
    def _is_numeric(self, value) -> bool:
        """Check if a value can be converted to numeric."""
        try:
            float(str(value).replace(',', '').replace('$', '').replace('%', ''))
            return True
        except (ValueError, TypeError):
            return False
    
    def _is_date(self, value) -> bool:
        """Check if a value can be converted to date."""
        try:
            pd.to_datetime(str(value))
            return True
        except (ValueError, TypeError):
            return False
    
    def _is_numeric_column(self, series: pd.Series) -> bool:
        """Check if a column should be numeric."""
        numeric_count = 0
        total_count = 0
        
        for value in series.dropna():
            total_count += 1
            if self._is_numeric(value):
                numeric_count += 1
        
        return total_count > 0 and (numeric_count / total_count) > 0.8
    
    def _is_date_column(self, series: pd.Series) -> bool:
        """Check if a column should be datetime."""
        date_count = 0
        total_count = 0
        
        for value in series.dropna():
            total_count += 1
            if self._is_date(value):
                date_count += 1
        
        return total_count > 0 and (date_count / total_count) > 0.8
    
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the dataframe."""
        cleaned_df = df.copy()
        
        # Remove completely empty rows and columns
        cleaned_df = cleaned_df.dropna(how='all')
        cleaned_df = cleaned_df.dropna(axis=1, how='all')
        
        # Clean column names
        cleaned_df.columns = [str(col).strip() for col in cleaned_df.columns]
        
        # Remove duplicate rows
        cleaned_df = cleaned_df.drop_duplicates()
        
        # Clean numeric columns
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                if self._is_numeric_column(cleaned_df[col]):
                    # Clean and convert to numeric
                    cleaned_df[col] = cleaned_df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
        
        return cleaned_df
    
    def validate_financial_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Specific validation for financial data."""
        validation_result = {
            "is_valid": True,
            "financial_metrics": {},
            "data_quality": {},
            "recommendations": []
        }
        
        # Check for required financial columns
        required_columns = ['revenue', 'expenses']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            validation_result["is_valid"] = False
            validation_result["recommendations"].append(
                f"Missing required columns: {', '.join(missing_columns)}"
            )
        
        # Calculate financial metrics
        if 'revenue' in df.columns and 'expenses' in df.columns:
            revenue = df['revenue'].sum()
            expenses = df['expenses'].sum()
            profit = revenue - expenses
            
            validation_result["financial_metrics"] = {
                "total_revenue": float(revenue),
                "total_expenses": float(expenses),
                "total_profit": float(profit),
                "profit_margin": float(profit / revenue) if revenue > 0 else 0
            }
            
            # Financial health checks
            if profit < 0:
                validation_result["recommendations"].append(
                    "Warning: Total expenses exceed total revenue"
                )
            
            if expenses > revenue * 0.9:
                validation_result["recommendations"].append(
                    "Warning: Expense ratio is very high (>90%)"
                )
        
        return validation_result


def validate_uploaded_file(file_content: bytes, filename: str) -> Dict[str, Any]:
    """Validate uploaded file before processing."""
    validation_result = {
        "is_valid": True,
        "file_type": None,
        "file_size": len(file_content),
        "errors": [],
        "warnings": []
    }
    
    # Check file size
    max_size = 50 * 1024 * 1024  # 50MB
    if len(file_content) > max_size:
        validation_result["is_valid"] = False
        validation_result["errors"].append(f"File size exceeds maximum limit of {max_size / (1024*1024):.0f}MB")
    
    # Check file extension
    allowed_extensions = ['.csv', '.xlsx', '.xls', '.pdf']
    file_ext = filename.lower().split('.')[-1] if '.' in filename else ''
    
    if f'.{file_ext}' not in allowed_extensions:
        validation_result["is_valid"] = False
        validation_result["errors"].append(f"Unsupported file type: {file_ext}")
    
    validation_result["file_type"] = file_ext
    
    return validation_result
