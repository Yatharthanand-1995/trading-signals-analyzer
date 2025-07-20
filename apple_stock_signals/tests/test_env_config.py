#!/usr/bin/env python3
"""
Unit tests for Environment Configuration
"""

import pytest
import os
import tempfile
from pathlib import Path

from config.env_config import EnvConfig, ConfigurationError

class TestEnvConfig:
    """
    Test suite for EnvConfig
    """
    
    def setup_method(self):
        """Set up test environment"""
        # Save current environment
        self.original_env = os.environ.copy()
        
        # Set test environment
        os.environ['ENVIRONMENT'] = 'test'
        os.environ['FINNHUB_API_KEY'] = 'test_key_123'
    
    def teardown_method(self):
        """Restore original environment"""
        # Clear all environment variables
        os.environ.clear()
        # Restore original
        os.environ.update(self.original_env)
    
    def test_initialization(self):
        """Test config initialization"""
        config = EnvConfig()
        assert config is not None
        assert config.ENVIRONMENT == 'test'
        assert config.FINNHUB_API_KEY == 'test_key_123'
    
    def test_api_key_access(self):
        """Test API key access"""
        config = EnvConfig()
        
        # Test with key set
        assert config.FINNHUB_API_KEY == 'test_key_123'
        
        # Test without key
        del os.environ['FINNHUB_API_KEY']
        config2 = EnvConfig()
        assert config2.FINNHUB_API_KEY is None
    
    def test_environment_detection(self):
        """Test environment detection"""
        # Test different environments
        environments = [
            ('development', True, False),
            ('staging', False, False),
            ('production', False, True),
            ('test', False, False)
        ]
        
        for env_name, is_dev, is_prod in environments:
            os.environ['ENVIRONMENT'] = env_name
            config = EnvConfig()
            
            assert config.ENVIRONMENT == env_name
            assert config.IS_DEVELOPMENT == is_dev
            assert config.IS_PRODUCTION == is_prod
    
    def test_risk_parameters(self):
        """Test risk management parameters"""
        config = EnvConfig()
        
        # Test defaults
        assert config.MAX_PORTFOLIO_RISK == 0.06
        assert config.MAX_POSITION_RISK == 0.02
        
        # Test custom values
        os.environ['MAX_PORTFOLIO_RISK'] = '0.08'
        os.environ['MAX_POSITION_RISK'] = '0.03'
        
        config2 = EnvConfig()
        assert config2.MAX_PORTFOLIO_RISK == 0.08
        assert config2.MAX_POSITION_RISK == 0.03
    
    def test_ml_configuration(self):
        """Test ML configuration"""
        config = EnvConfig()
        
        # Test defaults
        assert config.ENABLE_ML is False
        assert config.ML_MODEL_PATH == './ml_models/saved_models/'
        
        # Test enabling ML
        os.environ['ENABLE_ML'] = 'true'
        os.environ['ML_MODEL_PATH'] = '/custom/path/'
        
        config2 = EnvConfig()
        assert config2.ENABLE_ML is True
        assert config2.ML_MODEL_PATH == '/custom/path/'
    
    def test_database_configuration(self):
        """Test database configuration"""
        config = EnvConfig()
        
        # Test default
        assert config.DATABASE_URL == 'sqlite:///paper_trading.db'
        
        # Test custom database URL
        os.environ['DATABASE_URL'] = 'postgresql://user:pass@host:5432/db'
        config2 = EnvConfig()
        assert config2.DATABASE_URL == 'postgresql://user:pass@host:5432/db'
    
    def test_production_validation(self):
        """Test production environment validation"""
        os.environ['ENVIRONMENT'] = 'production'
        
        # Should fail without API key in production
        del os.environ['FINNHUB_API_KEY']
        
        with pytest.raises(ConfigurationError) as exc_info:
            config = EnvConfig()
        
        assert 'Missing required configuration' in str(exc_info.value)
    
    def test_safe_config_output(self):
        """Test safe configuration output (no secrets)"""
        os.environ['FINNHUB_API_KEY'] = 'super_secret_key_12345'
        os.environ['DATABASE_URL'] = 'postgresql://user:password@host:5432/db'
        
        config = EnvConfig()
        safe_config = config.get_safe_config()
        
        # Should not contain actual secrets
        assert 'super_secret_key_12345' not in str(safe_config)
        assert 'password' not in str(safe_config)
        
        # Should indicate presence of secrets
        assert safe_config['has_finnhub_key'] is True
        assert 'host:5432/db' in safe_config['database_url']
    
    def test_env_file_loading(self):
        """Test .env file loading"""
        # Create temporary .env file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write('TEST_VAR=test_value\n')
            f.write('FINNHUB_API_KEY=env_file_key\n')
            env_file = f.name
        
        try:
            # Temporarily change to temp directory
            original_dir = os.getcwd()
            temp_dir = os.path.dirname(env_file)
            os.chdir(temp_dir)
            
            # Rename to .env
            os.rename(env_file, os.path.join(temp_dir, '.env'))
            
            # Clear environment
            if 'TEST_VAR' in os.environ:
                del os.environ['TEST_VAR']
            if 'FINNHUB_API_KEY' in os.environ:
                del os.environ['FINNHUB_API_KEY']
            
            # Load config
            config = EnvConfig()
            
            # Should load from .env file
            assert os.environ.get('TEST_VAR') == 'test_value'
            assert config.FINNHUB_API_KEY == 'env_file_key'
            
        finally:
            # Cleanup
            os.chdir(original_dir)
            try:
                os.unlink(os.path.join(temp_dir, '.env'))
            except:
                pass
    
    def test_boolean_parsing(self):
        """Test boolean environment variable parsing"""
        test_cases = [
            ('true', True),
            ('True', True),
            ('TRUE', True),
            ('false', False),
            ('False', False),
            ('FALSE', False),
            ('0', False),
            ('1', False),  # Only 'true' is True
            ('yes', False),
            ('no', False)
        ]
        
        for value, expected in test_cases:
            os.environ['ENABLE_ML'] = value
            config = EnvConfig()
            assert config.ENABLE_ML == expected
    
    def test_monitoring_configuration(self):
        """Test monitoring configuration"""
        config = EnvConfig()
        
        # Test defaults
        assert config.ENABLE_MONITORING is False
        assert config.MONITORING_ENDPOINT is None
        
        # Test with monitoring enabled
        os.environ['ENABLE_MONITORING'] = 'true'
        os.environ['MONITORING_ENDPOINT'] = 'http://metrics.example.com'
        
        config2 = EnvConfig()
        assert config2.ENABLE_MONITORING is True
        assert config2.MONITORING_ENDPOINT == 'http://metrics.example.com'