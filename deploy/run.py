import uvicorn
import os
import sys

DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(DEPLOY_DIR)

for path in (PROJECT_ROOT, DEPLOY_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


def main():
    print("=" * 60)
    print("  Real Estate Price Predictor")
    print("=" * 60)
    print("\nЗапуск сервера...")
    print("Откройте в браузере: http://localhost:8000\n")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
