# 🧪 Testing Guide - Docker & CI/CD

This guide explains how the Docker container and CI/CD pipeline work, and how to test them.

---

## 📦 How Docker Container Works

### Current Setup
The Dockerfile creates a container that:
1. **Base Image**: Uses Python 3.11 slim (lightweight)
2. **Dependencies**: Installs all packages from `requirements.txt`
3. **Application**: Copies your entire project into `/app`
4. **Port**: Exposes port 8501 (Streamlit default)
5. **Health Check**: Monitors `/_stcore/health` endpoint
6. **Start Command**: Runs `streamlit run frontend/front.py`

### Container Lifecycle
```
Build → Run → Health Check → Serve Application
```

---

## 🚀 How to Test Locally

### Step 1: Build the Docker Image

```bash
# Build the image (this may take 5-10 minutes first time)
docker build -t floatchat:latest .

# Check if image was created
docker images | grep floatchat
```

**What happens:**
- Downloads Python 3.11 base image
- Installs system dependencies (build tools, curl)
- Installs all Python packages from `requirements.txt`
- Copies your project files
- Sets up the container environment

### Step 2: Run the Container

```bash
# Run the container (basic)
docker run --rm -p 8501:8501 floatchat:latest

# Run with environment variables from .env file
docker run --rm -p 8501:8501 --env-file .env floatchat:latest

# Run in detached mode (background)
docker run -d --name floatchat-app -p 8501:8501 --env-file .env floatchat:latest
```

**What happens:**
- Container starts and runs Streamlit
- Port 8501 is mapped from container to your host
- Application becomes available at `http://localhost:8501`

### Step 3: Verify It's Running

#### Check Container Status
```bash
# List running containers
docker ps

# Check container logs
docker logs floatchat-app

# Check container health
docker inspect floatchat-app | grep -A 10 Health
```

#### Test the Application
```bash
# Test health endpoint
curl http://localhost:8501/_stcore/health

# Open in browser
# Windows: start http://localhost:8501
# Mac/Linux: open http://localhost:8501
```

#### Stop the Container
```bash
# Stop detached container
docker stop floatchat-app

# Remove container
docker rm floatchat-app
```

---

## 🔄 How GitHub Actions CI/CD Works

### Workflow Triggers
The pipeline runs automatically when:
- ✅ You push code to `main` branch
- ✅ You create a Pull Request to `main` branch

### Pipeline Stages

#### Stage 1: Test Job
```yaml
test:
  - Checkout code
  - Setup Python 3.11
  - Install dependencies
  - Run static analysis (compileall)
```

**What it checks:**
- Python syntax errors
- Import errors
- Basic code structure

#### Stage 2: Docker Job (runs after test passes)
```yaml
docker:
  - Checkout code
  - Setup Docker Buildx
  - Login to GitHub Container Registry (only on main branch)
  - Build Docker image
  - Push to registry (only on main branch)
```

**What happens:**
- Builds the same Docker image
- On `main` branch: Pushes to `ghcr.io/your-username/your-repo:latest`
- On PR: Only builds (doesn't push)

---

## ✅ How to Check CI/CD Status

### Method 1: GitHub Web Interface

1. **Go to your repository** on GitHub
2. **Click "Actions" tab** (top navigation)
3. **View workflow runs:**
   - Green checkmark ✅ = Success
   - Red X ❌ = Failed
   - Yellow circle ⏳ = Running

4. **Click on a run** to see:
   - Which jobs ran
   - Logs from each step
   - Build time
   - Any errors

### Method 2: Check via Command Line

```bash
# Install GitHub CLI (if not installed)
# Windows: winget install GitHub.cli
# Mac: brew install gh

# Login to GitHub
gh auth login

# Check workflow status
gh run list

# View latest run
gh run view

# Watch a running workflow
gh run watch
```

### Method 3: Check Container Registry

After a successful push to `main`:

1. Go to your GitHub repository
2. Click **"Packages"** (right sidebar, or under repository name)
3. Find your container image
4. You'll see tags like `latest`

**Pull the image:**
```bash
docker pull ghcr.io/your-username/your-repo:latest
```

---

## 🐛 Troubleshooting

### Docker Build Fails

**Error: "Cannot find requirements.txt"**
```bash
# Make sure you're in project root
cd floatchat-clean
docker build -t floatchat:latest .
```

**Error: "Package installation fails"**
```bash
# Check if requirements.txt has correct packages
cat requirements.txt

# Try building with verbose output
docker build --progress=plain -t floatchat:latest .
```

**Error: "Port already in use"**
```bash
# Check what's using port 8501
# Windows: netstat -ano | findstr :8501
# Mac/Linux: lsof -i :8501

# Use a different port
docker run -p 8502:8501 floatchat:latest
# Then access at http://localhost:8502
```

### Container Runs But App Doesn't Load

**Check logs:**
```bash
docker logs floatchat-app
```

**Common issues:**
- Missing environment variables (check `.env` file)
- Database connection issues
- Port conflicts

### CI/CD Pipeline Fails

**Test job fails:**
- Check Python syntax in your code
- Verify `requirements.txt` is valid
- Check GitHub Actions logs for specific error

**Docker job fails:**
- Check Dockerfile syntax
- Verify all files are in repository
- Check if `.dockerignore` is excluding needed files

---

## 📊 Quick Test Checklist

- [ ] Docker image builds successfully
- [ ] Container starts without errors
- [ ] Application accessible at `http://localhost:8501`
- [ ] Health check endpoint responds
- [ ] GitHub Actions workflow runs on push
- [ ] Test job passes
- [ ] Docker job builds successfully
- [ ] Image pushed to registry (on main branch)

---

## 🎯 Next Steps

### For Local Development
1. Test the Docker build locally first
2. Run container and verify app works
3. Make changes and rebuild to test

### For CI/CD
1. Push code to a feature branch
2. Create Pull Request to `main`
3. Check Actions tab to see pipeline run
4. Merge PR to trigger image push
5. Pull and test the pushed image

### Enhancements You Can Add
- Add unit tests to the `test` job
- Add linting (flake8, black, pylint)
- Add security scanning
- Deploy to cloud (AWS, GCP, Azure)
- Add staging/production environments

---

## 📚 Useful Commands Reference

```bash
# Docker
docker build -t floatchat:latest .                    # Build image
docker run -p 8501:8501 floatchat:latest              # Run container
docker ps                                              # List running containers
docker logs <container-id>                             # View logs
docker stop <container-id>                             # Stop container
docker rm <container-id>                               # Remove container
docker images                                           # List images
docker rmi floatchat:latest                            # Remove image

# GitHub Actions
gh run list                                            # List workflow runs
gh run view                                            # View latest run
gh run watch                                           # Watch running workflow
gh workflow view CI                                    # View workflow details

# Testing
curl http://localhost:8501/_stcore/health              # Health check
docker exec -it <container-id> /bin/bash              # Enter container shell
```

---

## 💡 Understanding the Flow

```
Developer → Push Code → GitHub → Actions Triggered
                                    ↓
                            Test Job (Python checks)
                                    ↓
                            Docker Job (Build image)
                                    ↓
                            Push to Registry (if main branch)
                                    ↓
                            Image Available for Deployment
```

This setup ensures:
- ✅ Code quality checks before building
- ✅ Consistent builds across environments
- ✅ Automated image creation
- ✅ Ready for deployment

