# services/secrets_manager.py
"""
AWS Secrets Manager client with caching.
Used to fetch JWT secrets and API keys securely.
"""
import boto3
import json
import os
from functools import lru_cache
from typing import Any, Dict, Optional


class SecretsManager:
    """AWS Secrets Manager client with caching"""
    
    def __init__(self, region_name: str = "ap-south-1"):
        self.client = boto3.client(
            'secretsmanager',
            region_name=region_name
        )
        self._cache: Dict[str, Any] = {}
    
    @lru_cache(maxsize=10)
    def get_secret(self, secret_id: str) -> Dict[str, Any]:
        """
        Retrieve secret from AWS Secrets Manager.
        Cached with lru_cache to avoid repeated API calls.
        
        Args:
            secret_id: Secret name or ARN
            
        Returns:
            Dict containing parsed secret values
        """
        try:
            response = self.client.get_secret_value(SecretId=secret_id)
            secret_string = response['SecretString']
            return json.loads(secret_string)
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve secret {secret_id}: {str(e)}")
    
    def get_secret_value(self, secret_id: str, key: str, default: Optional[Any] = None) -> Any:
        """
        Get specific value from secret with fallback to default.
        
        Args:
            secret_id: Secret name or ARN
            key: Key within the secret JSON
            default: Default value if key not found
            
        Returns:
            Secret value or default
        """
        secret = self.get_secret(secret_id)
        return secret.get(key, default)


# ── Global Singleton Instance ─────────────────────────────────────────────────
secrets_manager = SecretsManager()


# ── Convenience Functions ─────────────────────────────────────────────────────

def get_jwt_secret() -> str:
    """
    Get JWT secret key from Secrets Manager (production) or .env (development).
    
    Returns:
        JWT secret key string
        
    Raises:
        RuntimeError: If secret not found and Secrets Manager disabled
    """
    use_secrets = os.getenv("USE_SECRETS_MANAGER", "false").lower() == "true"
    
    if use_secrets:
        secret_id = os.getenv("SECRET_ID_PATH", "wya/jwt-secret")
        return secrets_manager.get_secret_value(secret_id, "jwt-secret")
    else:
        # Fallback to .env for local development
        secret = os.getenv("SECRET_KEY")
        if not secret:
            raise RuntimeError(
                "SECRET_KEY not found in environment. "
                "Set USE_SECRETS_MANAGER=true on EC2 or add SECRET_KEY to .env locally."
            )
        return secret


def get_sagemaker_endpoint() -> str:
    """
    Get SageMaker endpoint name from Secrets Manager or .env.
    
    Returns:
        SageMaker endpoint name
    """
    use_secrets = os.getenv("USE_SECRETS_MANAGER", "false").lower() == "true"
    
    if use_secrets:
        secret_id = os.getenv("SECRET_ID_PATH", "wya/jwt-secret")
        return secrets_manager.get_secret_value(secret_id, "sagemaker-endpoint")
    else:
        return os.getenv("SAGEMAKER_ENDPOINT", "wya-fashionclip-serverless")


def get_vapid_keys() -> tuple[str, str]:
    """
    Get VAPID public and private keys from Secrets Manager or .env.
    
    Returns:
        Tuple of (public_key, private_key)
    """
    use_secrets = os.getenv("USE_SECRETS_MANAGER", "false").lower() == "true"
    
    if use_secrets:
        secret_id = os.getenv("SECRET_ID_PATH", "wya/jwt-secret")
        public_key = secrets_manager.get_secret_value(secret_id, "vapid-public-key")
        private_key = secrets_manager.get_secret_value(secret_id, "vapid-private-key")
        return public_key, private_key
    else:
        return (
            os.getenv("WYA_VAPID_PUBLIC_KEY"),
            os.getenv("WYA_VAPID_PRIVATE_KEY")
        )
