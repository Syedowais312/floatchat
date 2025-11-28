# PowerShell script to test Docker setup
# Run this script to verify your Docker configuration

Write-Host "🐳 FloatChat Docker Testing Script" -ForegroundColor Cyan
Write-Host "===================================`n" -ForegroundColor Cyan

# Check if Docker is running
Write-Host "1. Checking Docker installation..." -ForegroundColor Yellow
try {
    $dockerVersion = docker --version
    Write-Host "   ✅ Docker is installed: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "   ❌ Docker is not installed or not in PATH" -ForegroundColor Red
    exit 1
}

# Check if Docker daemon is running
Write-Host "`n2. Checking Docker daemon..." -ForegroundColor Yellow
try {
    docker ps | Out-Null
    Write-Host "   ✅ Docker daemon is running" -ForegroundColor Green
} catch {
    Write-Host "   ❌ Docker daemon is not running. Start Docker Desktop." -ForegroundColor Red
    exit 1
}

# Check if .env file exists
Write-Host "`n3. Checking environment file..." -ForegroundColor Yellow
if (Test-Path ".env") {
    Write-Host "   ✅ .env file found" -ForegroundColor Green
} else {
    Write-Host "   ⚠️  .env file not found (optional, but recommended)" -ForegroundColor Yellow
}

# Check if requirements.txt exists
Write-Host "`n4. Checking requirements.txt..." -ForegroundColor Yellow
if (Test-Path "requirements.txt") {
    Write-Host "   ✅ requirements.txt found" -ForegroundColor Green
} else {
    Write-Host "   ❌ requirements.txt not found!" -ForegroundColor Red
    exit 1
}

# Check if Dockerfile exists
Write-Host "`n5. Checking Dockerfile..." -ForegroundColor Yellow
if (Test-Path "Dockerfile") {
    Write-Host "   ✅ Dockerfile found" -ForegroundColor Green
} else {
    Write-Host "   ❌ Dockerfile not found!" -ForegroundColor Red
    exit 1
}

# Ask if user wants to build
Write-Host "`n6. Ready to build Docker image?" -ForegroundColor Yellow
$build = Read-Host "   Build image now? (y/n)"

if ($build -eq "y" -or $build -eq "Y") {
    Write-Host "`n   Building Docker image (this may take 5-10 minutes)..." -ForegroundColor Yellow
    docker build -t floatchat:latest .
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✅ Image built successfully!" -ForegroundColor Green
        Write-Host "`n   Next steps:" -ForegroundColor Cyan
        Write-Host "   1. Run: docker run --rm -p 8501:8501 --env-file .env floatchat:latest" -ForegroundColor White
        Write-Host "   2. Open: http://localhost:8501" -ForegroundColor White
    } else {
        Write-Host "   ❌ Build failed. Check errors above." -ForegroundColor Red
    }
} else {
    Write-Host "   Skipping build. Run manually with:" -ForegroundColor Yellow
    Write-Host "   docker build -t floatchat:latest ." -ForegroundColor White
}

Write-Host "`n✅ Testing complete!" -ForegroundColor Green

