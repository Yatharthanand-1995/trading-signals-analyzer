#!/usr/bin/env python3
"""
Environment Configuration Manager
Handles all environment variables and secrets securely
"""

import os
import sys
from typing import Dict, Any, Optional
from pathlib import Path
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing"""
    pass

class EnvConfig:
    """
    Centralized configuration management using environment variables
    """
    
    def __init__(self):
        self._load_env_file()
        self._validate_config()
        
    def _load_env_file(self):
        """Load .env file if it exists"""
        env_path = Path(__file__).parent.parent / '.env'
        
        if env_path.exists():
            try:
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            key, value = line.split('=', 1)
                            os.environ[key.strip()] = value.strip()
                logger.info("Loaded configuration from .env file")
            except Exception as e:
                logger.warning(f"Could not load .env file: {e}")
        else:
            logger.info("No .env file found, using system environment variables")
    
    def _validate_config(self):
        """Validate required configuration"""
        required_in_production = ['FINNHUB_API_KEY']
        
        if self.ENVIRONMENT == 'production':
            missing = []
            for key in required_in_production:
                if not getattr(self, key, None):
                    missing.append(key)
            
            if missing:
                raise ConfigurationError(
                    f"Missing required configuration for production: {', '.join(missing)}"
                )
    
    # API Keys
    @property
    def FINNHUB_API_KEY(self) -> Optional[str]:
        """Get Finnhub API key"""
        key = os.environ.get('FINNHUB_API_KEY')
        if not key and self.ENVIRONMENT != 'development':
            logger.warning("FINNHUB_API_KEY not set")
        return key
    
    @property
    def ALPHA_VANTAGE_API_KEY(self) -> Optional[str]:
        """Get Alpha Vantage API key"""
        return os.environ.get('ALPHA_VANTAGE_API_KEY')
    
    # Environment Settings
    @property
    def ENVIRONMENT(self) -> str:
        """Get current environment"""
        return os.environ.get('ENVIRONMENT', 'development').lower()
    
    @property
    def IS_PRODUCTION(self) -> bool:
        """Check if running in production"""
        return self.ENVIRONMENT == 'production'
    
    @property
    def IS_DEVELOPMENT(self) -> bool:
        """Check if running in development"""
        return self.ENVIRONMENT == 'development'
    
    @property
    def LOG_LEVEL(self) -> str:
        """Get log level"""
        return os.environ.get('LOG_LEVEL', 'INFO').upper()
    
    # Database Configuration
    @property
    def DATABASE_URL(self) -> str:
        """Get database URL"""
        return os.environ.get('DATABASE_URL', 'sqlite:///paper_trading.db')
    
    # Risk Management
    @property
    def MAX_PORTFOLIO_RISK(self) -> float:
        """Get maximum portfolio risk"""
        return float(os.environ.get('MAX_PORTFOLIO_RISK', '0.06'))
    
    @property
    def MAX_POSITION_RISK(self) -> float:
        """Get maximum position risk"""
        return float(os.environ.get('MAX_POSITION_RISK', '0.02'))
    
    # ML Configuration
    @property
    def ENABLE_ML(self) -> bool:
        """Check if ML is enabled"""
        return os.environ.get('ENABLE_ML', 'false').lower() == 'true'
    
    @property
    def ML_MODEL_PATH(self) -> str:
        """Get ML model path"""
        return os.environ.get('ML_MODEL_PATH', './ml_models/saved_models/')
    
    # Monitoring
    @property
    def ENABLE_MONITORING(self) -> bool:
        """Check if monitoring is enabled"""
        return os.environ.get('ENABLE_MONITORING', 'false').lower() == 'true'
    
    @property
    def MONITORING_ENDPOINT(self) -> Optional[str]:
        """Get monitoring endpoint"""
        return os.environ.get('MONITORING_ENDPOINT')
    
    def get_safe_config(self) -> Dict[str, Any]:
        """Get configuration safe for logging (no secrets)"""
        return {
            'environment': self.ENVIRONMENT,
            'log_level': self.LOG_LEVEL,
            'database_url': self.DATABASE_URL.split('@')[-1] if '@' in self.DATABASE_URL else self.DATABASE_URL,
            'max_portfolio_risk': self.MAX_PORTFOLIO_RISK,
            'max_position_risk': self.MAX_POSITION_RISK,
            'enable_ml': self.ENABLE_ML,
            'ml_model_path': self.ML_MODEL_PATH,
            'enable_monitoring': self.ENABLE_MONITORING,
            'has_finnhub_key': bool(self.FINNHUB_API_KEY),
            'has_alpha_vantage_key': bool(self.ALPHA_VANTAGE_API_KEY)
        }
    
    def __str__(self) -> str:
        """String representation (safe for logging)"""
        return json.dumps(self.get_safe_config(), indent=2)

# Singleton instance
config = EnvConfig()

# Export commonly used values
FINNHUB_API_KEY = config.FINNHUB_API_KEY
ENVIRONMENT = config.ENVIRONMENT
IS_PRODUCTION = config.IS_PRODUCTION
IS_DEVELOPMENT = config.IS_DEVELOPMENT
MAX_PORTFOLIO_RISK = config.MAX_PORTFOLIO_RISK
MAX_POSITION_RISK = config.MAX_POSITION_RISK

if __name__ == "__main__":
    print("Current Configuration:")
    print(config)
    
    if not config.FINNHUB_API_KEY and config.IS_PRODUCTION:
        print("\n⚠️  WARNING: No API key configured for production!")
    else:
        print("\n✅ Configuration loaded successfully")