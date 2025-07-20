# Security Configuration Guide

## Overview

This guide explains how to securely configure the trading system with proper credential management.

## Quick Start

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit the .env file with your credentials:**
   ```bash
   # Never commit this file to version control!
   FINNHUB_API_KEY=your_actual_api_key_here
   ALPHA_VANTAGE_API_KEY=your_actual_api_key_here
   ```

3. **Verify configuration:**
   ```bash
   python3 config/env_config.py
   ```

## Environment Variables

### Required Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|--------|
| `FINNHUB_API_KEY` | Finnhub API key for market data | Yes (in production) | 'demo' (dev only) |
| `ENVIRONMENT` | Current environment | No | 'development' |

### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ALPHA_VANTAGE_API_KEY` | Alpha Vantage API key | None |
| `DATABASE_URL` | Database connection string | 'sqlite:///paper_trading.db' |
| `LOG_LEVEL` | Logging level | 'INFO' |
| `MAX_PORTFOLIO_RISK` | Maximum portfolio risk | 0.06 (6%) |
| `MAX_POSITION_RISK` | Maximum position risk | 0.02 (2%) |
| `ENABLE_ML` | Enable ML features | false |
| `ML_MODEL_PATH` | Path to ML models | './ml_models/saved_models/' |
| `ENABLE_MONITORING` | Enable monitoring | false |
| `MONITORING_ENDPOINT` | Monitoring endpoint URL | None |

## Security Best Practices

### 1. Never Commit Secrets

- The `.env` file is in `.gitignore` and should NEVER be committed
- Use `.env.example` as a template for other developers
- Store production secrets in a secure secrets manager

### 2. Use Environment-Specific Files

```bash
# Development
.env                # Local development

# Production
.env.production     # Production environment (use secrets manager instead)
```

### 3. Validate Configuration

The system validates configuration on startup:

```python
from config.env_config import config

# Will raise ConfigurationError if required keys are missing in production
if config.IS_PRODUCTION and not config.FINNHUB_API_KEY:
    raise ConfigurationError("API key required in production")
```

### 4. Access Secrets Safely

```python
from config.env_config import config

# Good - uses environment variable
api_key = config.FINNHUB_API_KEY

# Bad - hardcoded secret
api_key = 'abc123'  # Never do this!
```

## Security Scanning

Run the security scanner to check for hardcoded secrets:

```bash
python3 scripts/security_check.py
```

This will scan all Python and configuration files for potential security issues.

## Production Deployment

### Using AWS Secrets Manager

```python
import boto3
import json

def get_secret(secret_name):
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

# In production
if config.IS_PRODUCTION:
    secrets = get_secret('trading-system/prod')
    os.environ['FINNHUB_API_KEY'] = secrets['finnhub_api_key']
```

### Using Docker

```dockerfile
# Dockerfile
FROM python:3.9

# Don't include .env in image
COPY . /app
RUN rm -f /app/.env

# Pass secrets at runtime
# docker run -e FINNHUB_API_KEY=$FINNHUB_API_KEY ...
```

### Using Kubernetes

```yaml
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: trading-secrets
type: Opaque
data:
  finnhub-api-key: <base64-encoded-key>

# deployment.yaml
env:
  - name: FINNHUB_API_KEY
    valueFrom:
      secretKeyRef:
        name: trading-secrets
        key: finnhub-api-key
```

## Troubleshooting

### Configuration Not Loading

1. Check if `.env` file exists
2. Verify file permissions: `chmod 600 .env`
3. Check for syntax errors in `.env`
4. Run `python3 config/env_config.py` to debug

### API Key Issues

1. Verify key is set: `echo $FINNHUB_API_KEY`
2. Check if key is valid with provider
3. Ensure no extra spaces in `.env` file
4. Check rate limits with API provider

## Security Checklist

- [ ] `.env` file created and not in version control
- [ ] All hardcoded secrets removed from code
- [ ] Security scan passes (`python3 scripts/security_check.py`)
- [ ] Production uses secrets manager, not `.env` files
- [ ] API keys have minimum required permissions
- [ ] Regular key rotation schedule in place
- [ ] Monitoring for unauthorized API usage
- [ ] Logs don't contain sensitive information

## Summary

Proper security configuration is critical for protecting your trading system and API keys. Always:

1. Use environment variables for all secrets
2. Never commit secrets to version control
3. Use different keys for development and production
4. Implement key rotation
5. Monitor for security issues

For questions or issues, run the security check script or review the `config/env_config.py` module.