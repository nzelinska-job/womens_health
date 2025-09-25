import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)


class GitHubConfig:
    """GitHub repository configuration"""

    def __init__(self):
        # Try Streamlit secrets first, then environment variables
        self.token = self._get_config_value("github_token", "GITHUB_TOKEN")
        self.owner = self._get_config_value(
            "data_sources.repo_owner", "GITHUB_OWNER", "nzelinska-job")
        self.repo = self._get_config_value(
            "data_sources.repo_name", "GITHUB_REPO", "cycle-tracker-data")
        self.branch = self._get_config_value(
            "data_sources.branch", "GITHUB_BRANCH", "main")

    def _get_config_value(self, streamlit_key, env_key, default=None):
        """Get configuration value from Streamlit secrets or environment variables"""
        try:
            # Try Streamlit secrets first
            if "." in streamlit_key:
                keys = streamlit_key.split(".")
                value = st.secrets
                for key in keys:
                    value = value.get(key, {})
                if value and value != {}:
                    return value
            else:
                if streamlit_key in st.secrets:
                    return st.secrets[streamlit_key]
        except:
            pass

        # Fall back to environment variables
        return os.getenv(env_key, default)

    @property
    def headers(self):
        """Get authentication headers for GitHub API"""
        if self.token:
            return {"Authorization": f"token {self.token}"}
        return {}

    @property
    def base_raw_url(self):
        """Get base URL for raw file access"""
        return f"https://raw.githubusercontent.com/{self.owner}/{self.repo}/{self.branch}"

    @property
    def base_api_url(self):
        """Get base URL for GitHub API access"""
        return f"https://api.github.com/repos/{self.owner}/{self.repo}/contents"


class AppConfig:
    """Application configuration"""

    def __init__(self):
        self.default_cycle_length = int(self._get_config_value(
            "app_settings.default_cycle_length", "DEFAULT_CYCLE_LENGTH", "28"))
        self.min_cycle_length = int(self._get_config_value(
            "app_settings.min_cycle_length", "MIN_CYCLE_LENGTH", "21"))
        self.max_cycle_length = int(self._get_config_value(
            "app_settings.max_cycle_length", "MAX_CYCLE_LENGTH", "35"))
        self.environment = self._get_config_value(
            "environment", "ENVIRONMENT", "production")
        self.debug = self._get_config_value(
            "debug", "DEBUG", "false").lower() == "true"

    def _get_config_value(self, streamlit_key, env_key, default=None):
        """Get configuration value from Streamlit secrets or environment variables"""
        try:
            if "." in streamlit_key:
                keys = streamlit_key.split(".")
                value = st.secrets
                for key in keys:
                    value = value.get(key, {})
                if value and value != {}:
                    return str(value)
            else:
                if streamlit_key in st.secrets:
                    return str(st.secrets[streamlit_key])
        except:
            pass

        return os.getenv(env_key, default)


# Global configuration instances
github_config = GitHubConfig()
app_config = AppConfig()

# File URLs for easy access
USERS_CSV_URL = f"{github_config.base_raw_url}/users.csv"
RECOMMENDATIONS_JSON_URL = f"{github_config.base_raw_url}/recommendations.json"
MODEL_PICKLE_URL = f"{github_config.base_raw_url}/cycle_model.pkl"

# API URLs
USERS_API_URL = f"{github_config.base_api_url}/users.csv"
RECOMMENDATIONS_API_URL = f"{github_config.base_api_url}/recommendations.json"
MODEL_API_URL = f"{github_config.base_api_url}/cycle_model.pkl"


def validate_configuration():
    """Validate that all required configuration is present"""
    errors = []

    if not github_config.token:
        errors.append(
            "GitHub token is required. Set GITHUB_TOKEN environment variable or add to secrets.toml")

    if not github_config.owner:
        errors.append("GitHub owner/username is required")

    if not github_config.repo:
        errors.append("GitHub repository name is required")

    if errors:
        return False, errors

    return True, []
