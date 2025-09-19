from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.schemas import KaggleHouseFeatures, PredictionResponse
from app.service.model import get_model_path, is_model_loaded, load_model, predict

app = FastAPI(title="House Price Prediction API", version="0.2.0")

# Enable CORS for local development and simple frontend integrations
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# Mount separate frontend directory (static files)
app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")


@app.on_event("startup")
def startup_event() -> None:
	# Try to load model on startup; if not present, keep app running with health info
	try:
		load_model()
	except FileNotFoundError:
		# Model not trained yet; endpoints will report accordingly
		pass


@app.get("/", response_class=HTMLResponse)
def index() -> str:
	return (
		"""
		<!doctype html>
		<html>
		<head><meta charset=\"utf-8\"><title>House Price Prediction API</title></head>
		<body style=\"font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 2rem;\">
			<h1>House Price Prediction API</h1>
			<p>This is a backend API built with FastAPI.</p>
			<ul>
				<li><a href=\"/docs\">Open API Docs (Swagger UI)</a></li>
				<li><a href=\"/health\">Health Check</a></li>
				<li><a href=\"/frontend/index.html\">Simple Frontend</a></li>
			</ul>
			<p>Use <code>GET /predict</code> with query parameters to get predictions. See the docs for schema.</p>
		</body>
		</html>
		"""
	)


@app.get("/health")
def health() -> JSONResponse:
	status = {
		"status": "ok",
		"model_loaded": is_model_loaded(),
		"model_path": get_model_path(),
	}
	return JSONResponse(status)


@app.post("/predict", response_model=PredictionResponse)
def predict_price(body: KaggleHouseFeatures) -> PredictionResponse:
    # Lazy auto-load: if not loaded, attempt to load now
    if not is_model_loaded():
        try:
            load_model()
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e))

    pred_value = predict(body.features)
    return PredictionResponse(price=pred_value)


@app.post("/admin/reload-model")
def admin_reload_model() -> JSONResponse:
	try:
		load_model()
		return JSONResponse({"reloaded": True, "model_path": get_model_path()})
	except FileNotFoundError as e:
		raise HTTPException(status_code=404, detail=str(e))
