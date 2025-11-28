#!/bin/bash
# Bash script to test Docker setup
# Run this script to verify your Docker configuration

echo "🐳 FloatChat Docker Testing Script"
echo "==================================="
echo ""

# Check if Docker is running
echo "1. Checking Docker installation..."
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version)
    echo "   ✅ Docker is installed: $DOCKER_VERSION"
else
    echo "   ❌ Docker is not installed or not in PATH"
    exit 1
fi

# Check if Docker daemon is running
echo ""
echo "2. Checking Docker daemon..."
if docker ps &> /dev/null; then
    echo "   ✅ Docker daemon is running"
else
    echo "   ❌ Docker daemon is not running. Start Docker Desktop."
    exit 1
fi

# Check if .env file exists
echo ""
echo "3. Checking environment file..."
if [ -f ".env" ]; then
    echo "   ✅ .env file found"
else
    echo "   ⚠️  .env file not found (optional, but recommended)"
fi

# Check if requirements.txt exists
echo ""
echo "4. Checking requirements.txt..."
if [ -f "requirements.txt" ]; then
    echo "   ✅ requirements.txt found"
else
    echo "   ❌ requirements.txt not found!"
    exit 1
fi

# Check if Dockerfile exists
echo ""
echo "5. Checking Dockerfile..."
if [ -f "Dockerfile" ]; then
    echo "   ✅ Dockerfile found"
else
    echo "   ❌ Dockerfile not found!"
    exit 1
fi

# Ask if user wants to build
echo ""
echo "6. Ready to build Docker image?"
read -p "   Build image now? (y/n): " build

if [ "$build" = "y" ] || [ "$build" = "Y" ]; then
    echo ""
    echo "   Building Docker image (this may take 5-10 minutes)..."
    docker build -t floatchat:latest .
    
    if [ $? -eq 0 ]; then
        echo "   ✅ Image built successfully!"
        echo ""
        echo "   Next steps:"
        echo "   1. Run: docker run --rm -p 8501:8501 --env-file .env floatchat:latest"
        echo "   2. Open: http://localhost:8501"
    else
        echo "   ❌ Build failed. Check errors above."
    fi
else
    echo "   Skipping build. Run manually with:"
    echo "   docker build -t floatchat:latest ."
fi

echo ""
echo "✅ Testing complete!"

