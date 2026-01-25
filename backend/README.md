# Backend - Paper Trading API

FastAPI backend for the paper trading system.

## Setup

1. Create a virtual environment (recommended):
```bash
python3 -m venv venv
```

2. Activate the virtual environment:
```bash
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running

```bash
# Using 'python -m uvicorn' ensures uvicorn runs from the activated venv
# This prevents "command not found" errors if uvicorn isn't in your system PATH
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Why use `python -m uvicorn` instead of just `uvicorn`?**
- Ensures uvicorn runs from the activated virtual environment
- Prevents "command not found" errors if uvicorn isn't in your system PATH
- Works consistently across different operating systems

The API will be available at `http://localhost:8000`

API docs available at `http://localhost:8000/docs`
