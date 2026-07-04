set -e

if ! command -v brew &> /dev/null; then
    echo "Homebrew not found. Install it from https://brew.sh, then re-run this script."
    exit 1
fi

echo "Installing libomp (required by XGBoost & LightGBM on macOS)..."
brew install libomp

echo "Creating virtual environment (./venv)..."
python3 -m venv venv
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "Setup complete."
echo "Activate the environment in future sessions with:"
echo "    source venv/bin/activate"
