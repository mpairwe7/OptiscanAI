# Docker Deployment Scripts

This directory contains scripts for building and deploying the retinal disease screening application using Podman and Docker Hub.

## Prerequisites

- Podman installed
- Docker Hub account
- Application code and dependencies

## Environment Variables

Set these environment variables for secure Docker Hub authentication:

```bash
export DOCKERHUB_USERNAME=your_dockerhub_username
export DOCKERHUB_PASSWORD=your_dockerhub_password
```

## Build Scripts

### GPU Version
```bash
# Build GPU-enabled image
./podman_build.sh

# Build CPU-only image
./podman_build_cpu.sh
```

### Push Scripts

#### GPU Version
```bash
# Set credentials (or export them)
export DOCKERHUB_USERNAME=your_username
export DOCKERHUB_PASSWORD=your_password

# Push GPU image to Docker Hub
./podman_push.sh
```

#### CPU Version
```bash
# Set credentials (or export them)
export DOCKERHUB_USERNAME=your_username
export DOCKERHUB_PASSWORD=your_password

# Push CPU image to Docker Hub
./podman_push_cpu.sh
```

## Automatic Versioning

The build scripts automatically generate version tags based on:

1. **Git Tags**: If a git tag exists (e.g., `v2.1.0`), uses `gpu-v2.1.0` or `cpu-v2.1.0`
2. **Git Commit**: If no tag, uses commit hash: `gpu-v2.1.0-d17afce` or `cpu-v2.1.0-d17afce`
3. **Fallback**: Static version if git not available: `gpu-v2.1.0` or `cpu-v2.1.0`

## Image Tags

### GPU Images
- `your_username/retinal-screening:gpu-v2.1.0` (semantic version)
- `your_username/retinal-screening:gpu-v2.1.0-d17afce` (with commit hash)
- `your_username/retinal-screening:latest-gpu` (latest GPU)
- `your_username/retinal-screening:20241209_143022` (timestamped)

### CPU Images
- `your_username/retinal-screening:cpu-v2.1.0` (semantic version)
- `your_username/retinal-screening:cpu-v2.1.0-d17afce` (with commit hash)
- `your_username/retinal-screening:latest-cpu` (latest CPU)
- `your_username/retinal-screening:20241209_143022` (timestamped)

## Security Notes

- Never commit credentials to version control
- Use environment variables or Docker Hub access tokens
- Consider using Docker Hub's automated builds for CI/CD pipelines

## Deployment

After pushing to Docker Hub, you can deploy from anywhere:

```bash
# Run GPU version
podman run -p 8080:8080 -p 8501:8501 your_username/retinal-screening:latest-gpu

# Run CPU version
podman run -p 8080:8080 -p 8501:8501 your_username/retinal-screening:latest-cpu
```

## Troubleshooting

- Ensure Podman is logged into Docker Hub: `podman login docker.io`
- Check image exists: `podman images | grep retinal-screening`
- Verify tags: `podman images your_username/retinal-screening`