.PHONY: frontend backend

frontend:
	cd frontend && npm run dev

backend:
	.venv/bin/uvicorn main:app --reload --host 0.0.0.0 --port 8000
