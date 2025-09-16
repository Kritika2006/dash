from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
import camelot
import tempfile
import os

from ..core.schemas import (
    AskPayload,
    AnalysisResult,
    TimeSeriesData,
    ForecastRequest,
    MonteCarloRequest,
    StressTestRequest,
    ESGRequest,
)
from ..services import analysis_service, llm_service, forecasting_service
from ..services import risk_service, data_validation_service


router = APIRouter()

@router.post("/process-file", response_model=AnalysisResult)
async def process_file_for_dashboard(file: UploadFile = File(...)):
    """
    Processes CSV, Excel, or a complex PDF with a robust, multi-pass strategy.
    """
    # Validate file before processing
    file_content = await file.read()
    file_validation = data_validation_service.validate_uploaded_file(file_content, file.filename)
    
    if not file_validation["is_valid"]:
        raise HTTPException(
            status_code=400, 
            detail=f"File validation failed: {'; '.join(file_validation['errors'])}"
        )
    
    df = None
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(file_content)
        tmp_path = tmp.name

    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(tmp_path)
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(tmp_path)
        elif file.filename.endswith('.pdf'):
            print("Attempting to parse PDF with a multi-step strategy...")
            df = None
            
            # --- Robust PDF Parsing Logic ---
            # Scan through the most likely pages for a usable table
            for page_num in range(4, 26): # Scan pages 4-25
                try:
                    # Strategy 1: Try 'lattice' first, it's cleaner for tables with lines
                    tables = camelot.read_pdf(tmp_path, pages=str(page_num), flavor='lattice', line_scale=40)
                    if tables.n > 0 and tables[0].df.shape[0] > 3 and tables[0].df.shape[1] > 3:
                        df = tables[0].df
                        print(f"Success! Found a table on page {page_num} using 'lattice'.")
                        break  # Exit the loop once a good table is found

                    # Strategy 2: If lattice fails, try 'stream'
                    tables = camelot.read_pdf(tmp_path, pages=str(page_num), flavor='stream')
                    if tables.n > 0 and tables[0].df.shape[0] > 3 and tables[0].df.shape[1] > 3:
                        df = tables[0].df
                        print(f"Success! Found a table on page {page_num} using 'stream'.")
                        break  # Exit the loop
                except Exception:
                    # Ignore errors on individual pages (e.g., blank pages) and continue
                    continue
            
            if df is None:
                raise HTTPException(status_code=400, detail="Failed to find any suitable tables in the PDF after scanning multiple pages.")

            # --- Generic Data Cleaning for the extracted table ---
            df.replace(r'^\s*$', pd.NA, regex=True, inplace=True)
            header_row_index = 0
            for i, row in df.iterrows():
                if row.notna().sum() > 2: # Find first row with at least 3 non-empty cells to use as header
                    header_row_index = i
                    break
            
            df.columns = df.iloc[header_row_index]
            df = df.iloc[header_row_index + 1:].reset_index(drop=True)
            df.dropna(how='all', inplace=True) # Drop rows that are completely empty
            print("Successfully extracted and cleaned a table from the PDF.")
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        # Validate and clean the dataframe
        validator = data_validation_service.DataValidator()
        validation_result = validator.validate_dataframe(df)
        
        if not validation_result["is_valid"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Data validation failed: {'; '.join(validation_result['errors'])}"
            )
        
        # Clean the dataframe
        df = validator.clean_dataframe(df)
        
        # Prepare response pieces
        df.columns = df.columns.astype(str)
        financial_input, expense_data, runway_data = analysis_service.process_dataframe(df)
        kpis = analysis_service.calculate_kpis(financial_input)
        advisory = analysis_service.generate_advisory(kpis)
        df.fillna(0, inplace=True)
        data_preview = df.to_dict(orient='records')
        # Risk
        risk_factors = risk_service.compute_risk_from_financials(
            revenue=financial_input.revenue,
            expenses=financial_input.expenses,
            liabilities=financial_input.liabilities,
            cash=financial_input.cash or 0.0,
        )
        risk_summary = risk_service.summarize_risk(risk_factors)

        return AnalysisResult(
            kpis=kpis,
            advisory=advisory,
            expense_chart_data=expense_data,
            runway_chart_data=runway_data,
            data_preview=data_preview,
            risk=risk_summary,
        )

    except Exception as e:
        print(f"An error occurred in process_file_for_dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(tmp_path)

@router.post("/risk")
async def compute_risk(file: UploadFile = File(...)):
    """Compute risk directly from a simple CSV/XLSX file."""
    df = None
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(tmp_path)
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(tmp_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        return risk_service.compute_risk_from_dataframe(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(tmp_path)

@router.post("/ask")
def ask_ai(payload: AskPayload):
    """Endpoint for the AI chatbot."""
    answer = llm_service.llm_instance.ask(payload)
    return {"answer": answer}

# cfo_dashboard/backend/api/router.py
# Add this import at the top
from ..services import forecasting_service

# ... (keep your existing router and the /process-file and /ask endpoints) ...

# cfo_dashboard/backend/api/router.py
# Make sure to import the new schema at the top
from ..core.schemas import AskPayload, AnalysisResult, TimeSeriesData
# ... (keep other imports and endpoints) ...

@router.post("/forecast")
async def get_forecast(data: ForecastRequest):
    """Accepts cleaned records and returns a forecast for next 12 months."""
    try:
        df = pd.DataFrame(data.records)
        date_col = df.columns[0]
        value_col = df.columns[1]
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        df.dropna(subset=[date_col], inplace=True)
        df.set_index(date_col, inplace=True)
        series_to_forecast = df[value_col].resample('MS').sum()

        forecast_df = forecasting_service.generate_forecast(
            series_to_forecast, forecast_steps=12, model=data.model
        )

        historical_df = series_to_forecast.reset_index()
        historical_df.columns = ['date', 'actual']

        forecast_df.reset_index(inplace=True)
        forecast_df.rename(columns={'index': 'date'}, inplace=True)

        return {"historical": historical_df.to_dict(orient='records'), "forecast": forecast_df.to_dict(orient='records')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating forecast: {e}")


@router.post("/monte-carlo")
async def monte_carlo_simulation(request: MonteCarloRequest):
    """Run Monte Carlo simulation for cash flow forecasting."""
    try:
        result = risk_service.monte_carlo_cash_flow_simulation(
            initial_cash=request.initial_cash,
            monthly_revenue_mean=request.monthly_revenue_mean,
            monthly_revenue_std=request.monthly_revenue_std,
            monthly_expenses_mean=request.monthly_expenses_mean,
            monthly_expenses_std=request.monthly_expenses_std,
            months=request.months,
            simulations=request.simulations
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in Monte Carlo simulation: {e}")


@router.post("/stress-test")
async def stress_test(request: StressTestRequest):
    """Run stress test scenarios."""
    try:
        result = risk_service.stress_test_scenarios(
            base_revenue=request.base_revenue,
            base_expenses=request.base_expenses,
            base_cash=request.base_cash,
            scenarios=request.scenarios
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in stress test: {e}")


@router.post("/esg-score")
async def calculate_esg_score(request: ESGRequest):
    """Calculate ESG risk score."""
    try:
        result = risk_service.calculate_esg_risk_score(
            environmental_score=request.environmental_score,
            social_score=request.social_score,
            governance_score=request.governance_score
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating ESG score: {e}")


@router.post("/financial-ratios")
async def calculate_financial_ratios(revenue: float, expenses: float, liabilities: float, cash: float):
    """Calculate comprehensive financial ratios."""
    try:
        result = risk_service.calculate_financial_ratios(revenue, expenses, liabilities, cash)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating financial ratios: {e}")


@router.post("/multiple-forecasts")
async def get_multiple_forecasts(data: ForecastRequest):
    """Generate forecasts using multiple models for comparison."""
    try:
        df = pd.DataFrame(data.records)
        date_col = df.columns[0]
        value_col = df.columns[1]
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        df.dropna(subset=[date_col], inplace=True)
        df.set_index(date_col, inplace=True)
        series_to_forecast = df[value_col].resample('MS').sum()

        forecasts = forecasting_service.generate_multiple_forecasts(series_to_forecast, forecast_steps=12)
        
        # Convert forecasts to serializable format
        result = {}
        for model_name, forecast_df in forecasts.items():
            if forecast_df is not None:
                forecast_df.reset_index(inplace=True)
                forecast_df.rename(columns={'index': 'date'}, inplace=True)
                result[model_name] = forecast_df.to_dict(orient='records')
            else:
                result[model_name] = None

        historical_df = series_to_forecast.reset_index()
        historical_df.columns = ['date', 'actual']
        
        return {
            "historical": historical_df.to_dict(orient='records'),
            "forecasts": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating multiple forecasts: {e}")


@router.post("/validate-data")
async def validate_data(file: UploadFile = File(...)):
    """Validate uploaded data for quality and consistency."""
    try:
        file_content = await file.read()
        file_validation = data_validation_service.validate_uploaded_file(file_content, file.filename)
        
        if not file_validation["is_valid"]:
            return file_validation
        
        # Process the file to get dataframe
        df = None
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name
        
        try:
            if file.filename.endswith('.csv'):
                df = pd.read_csv(tmp_path)
            elif file.filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(tmp_path)
            else:
                return {"error": "Unsupported file type for validation"}
            
            # Validate dataframe
            validator = data_validation_service.DataValidator()
            validation_result = validator.validate_dataframe(df)
            financial_validation = validator.validate_financial_data(df)

            return {
                "file_validation": file_validation,
                "dataframe_validation": validation_result,
                "financial_validation": financial_validation
            }
            
        finally:
            os.remove(tmp_path)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error validating data: {e}")


@router.get("/load-sample-data", response_model=AnalysisResult)
async def load_sample_data():
    """Load the sample Zomato PDF data."""
    try:
        # Path to the sample PDF
        sample_pdf_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "Zomato_Annual_Report_2023-24.pdf")
        
        if not os.path.exists(sample_pdf_path):
            raise HTTPException(status_code=404, detail="Sample PDF not found")
        
        # Process the PDF directly
        df = None
        print("Attempting to parse sample PDF with a multi-step strategy...")
        
        # --- Robust PDF Parsing Logic ---
        # Scan through the most likely pages for a usable table
        for page_num in range(4, 26): # Scan pages 4-25
            try:
                # Strategy 1: Try 'lattice' first, it's cleaner for tables with lines
                tables = camelot.read_pdf(sample_pdf_path, pages=str(page_num), flavor='lattice', line_scale=40)
                if tables.n > 0 and tables[0].df.shape[0] > 3 and tables[0].df.shape[1] > 3:
                    df = tables[0].df
                    print(f"Success! Found a table on page {page_num} using 'lattice'.")
                    break  # Exit the loop once a good table is found

                # Strategy 2: If lattice fails, try 'stream'
                tables = camelot.read_pdf(sample_pdf_path, pages=str(page_num), flavor='stream')
                if tables.n > 0 and tables[0].df.shape[0] > 3 and tables[0].df.shape[1] > 3:
                    df = tables[0].df
                    print(f"Success! Found a table on page {page_num} using 'stream'.")
                    break  # Exit the loop
            except Exception:
                # Ignore errors on individual pages (e.g., blank pages) and continue
                continue
        
        if df is None:
            raise HTTPException(status_code=400, detail="Failed to find any suitable tables in the sample PDF")

        # --- Generic Data Cleaning for the extracted table ---
        df.replace(r'^\s*$', pd.NA, regex=True, inplace=True)
        header_row_index = 0
        for i, row in df.iterrows():
            if row.notna().sum() > 2: # Find first row with at least 3 non-empty cells to use as header
                header_row_index = i
                break
        
        df.columns = df.iloc[header_row_index]
        df = df.iloc[header_row_index + 1:].reset_index(drop=True)
        df.dropna(how='all', inplace=True) # Drop rows that are completely empty
        print("Successfully extracted and cleaned a table from the sample PDF.")

        # Validate and clean the dataframe
        validator = data_validation_service.DataValidator()
        validation_result = validator.validate_dataframe(df)
        
        if not validation_result["is_valid"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Data validation failed: {'; '.join(validation_result['errors'])}"
            )
        
        # Clean the dataframe
        df = validator.clean_dataframe(df)
        
        # Prepare response pieces
        df.columns = df.columns.astype(str)
        financial_input, expense_data, runway_data = analysis_service.process_dataframe(df)
        kpis = analysis_service.calculate_kpis(financial_input)
        advisory = analysis_service.generate_advisory(kpis)
        df.fillna(0, inplace=True)
        data_preview = df.to_dict(orient='records')
        
        # Risk analysis
        risk_factors = risk_service.compute_risk_from_financials(
            revenue=financial_input.revenue,
            expenses=financial_input.expenses,
            liabilities=financial_input.liabilities,
            cash=financial_input.cash or 0.0,
        )
        risk_summary = risk_service.summarize_risk(risk_factors)

        return AnalysisResult(
            kpis=kpis,
            advisory=advisory,
            expense_chart_data=expense_data,
            runway_chart_data=runway_data,
            data_preview=data_preview,
            risk=risk_summary,
        )

    except Exception as e:
        print(f"An error occurred in load_sample_data: {e}")
        raise HTTPException(status_code=500, detail=str(e))