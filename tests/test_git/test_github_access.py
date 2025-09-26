# tests/test_github_access.py
"""
Test suite for GitHub private repository access
Tests both raw URLs and GitHub API endpoints for the cycle-tracker-data repository
"""

import pytest
import requests
import json
import base64
import os
from unittest.mock import patch, MagicMock
import pandas as pd
import io

# Configuration for the specific repository
REPO_OWNER = "nzelinska-job"
REPO_NAME = "cycle-tracker-data"
BRANCH = "main"

# File paths in the repository
FILES_TO_TEST = ["users.csv", "recommendations.json", "cycle_model.pkl"]


class TestGitHubAccess:
    """Test class for GitHub repository access functionality"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        self.repo_owner = REPO_OWNER
        self.repo_name = REPO_NAME
        self.branch = BRANCH
        self.github_token = os.getenv("GITHUB_TOKEN")

        # URLs for testing
        self.base_raw_url = f"https://raw.githubusercontent.com/{self.repo_owner}/{self.repo_name}/{self.branch}"
        self.base_api_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/contents"

        self.headers = {}
        if self.github_token:
            self.headers = {"Authorization": f"token {self.github_token}"}

    def test_github_token_exists(self):
        """Test that GitHub token is available for authentication"""
        assert self.github_token is not None, (
            "GitHub token not found. Set GITHUB_TOKEN environment variable or "
            "add github_token to .streamlit/secrets.toml"
        )
        assert self.github_token.startswith("ghp_"), (
            f"Invalid GitHub token format. Token should start with 'ghp_', "
            f"got: {self.github_token[:10]}..."
        )
        assert len(self.github_token) >= 36, (
            f"GitHub token too short. Expected at least 36 characters, "
            f"got: {len(self.github_token)}"
        )

    def test_private_repo_requires_authentication(self):
        """Test that private repository returns 404 without authentication"""
        url = f"{self.base_raw_url}/users.csv"
        response = requests.get(url)

        assert response.status_code == 404, (
            f"Expected 404 for unauthenticated request to private repo, "
            f"got: {response.status_code}. Repository might be public or URL is wrong."
        )

    @pytest.mark.parametrize("filename", FILES_TO_TEST)
    def test_raw_url_access_with_token(self, filename):
        """Test raw URL access with GitHub token for each file"""
        if not self.github_token:
            pytest.skip("GitHub token not available")

        url = f"{self.base_raw_url}/{filename}"
        response = requests.get(url, headers=self.headers)

        if filename == "cycle_model.pkl":
            # Model file might not exist, so 404 is acceptable
            assert response.status_code in [200, 404], (
                f"Expected 200 or 404 for {filename}, got: {response.status_code}\n"
                f"URL: {url}\n"
                f"Response: {response.text[:200]}"
            )
        else:
            assert response.status_code == 200, (
                f"Failed to access {filename} via raw URL. "
                f"Status: {response.status_code}\n"
                f"URL: {url}\n"
                f"Response: {response.text[:200]}"
            )

            # Validate content type
            if filename.endswith('.csv'):
                assert 'text' in response.headers.get('content-type', ''), (
                    f"Expected text content for CSV file, got: "
                    f"{response.headers.get('content-type')}"
                )
            elif filename.endswith('.json'):
                # Test if JSON is valid
                try:
                    json.loads(response.text)
                except json.JSONDecodeError as e:
                    pytest.fail(f"Invalid JSON in {filename}: {e}")

    @pytest.mark.parametrize("filename", FILES_TO_TEST)
    def test_api_url_access_with_token(self, filename):
        """Test GitHub API access with token for each file"""
        if not self.github_token:
            pytest.skip("GitHub token not available")

        url = f"{self.base_api_url}/{filename}"
        response = requests.get(url, headers=self.headers)

        if filename == "cycle_model.pkl":
            # Model file might not exist
            assert response.status_code in [200, 404], (
                f"Expected 200 or 404 for {filename} via API, got: {response.status_code}"
            )
        else:
            assert response.status_code == 200, (
                f"Failed to access {filename} via GitHub API. "
                f"Status: {response.status_code}\n"
                f"URL: {url}\n"
                f"Response: {response.text[:200]}"
            )

            if response.status_code == 200:
                data = response.json()

                # Validate API response structure
                assert 'content' in data, f"API response missing 'content' field for {filename}"
                assert 'encoding' in data, f"API response missing 'encoding' field for {filename}"
                assert data['encoding'] == 'base64', (
                    f"Expected base64 encoding for {filename}, got: {data['encoding']}"
                )

                # Test content decoding
                try:
                    decoded_content = base64.b64decode(
                        data['content']).decode('utf-8')
                    assert len(
                        decoded_content) > 0, f"Decoded content is empty for {filename}"
                except Exception as e:
                    if filename != "cycle_model.pkl":  # Binary files may not decode as UTF-8
                        pytest.fail(
                            f"Failed to decode content for {filename}: {e}")

    def test_users_csv_structure(self):
        """Test that users.csv has the expected structure"""
        if not self.github_token:
            pytest.skip("GitHub token not available")

        url = f"{self.base_raw_url}/users.csv"
        response = requests.get(url, headers=self.headers)

        assert response.status_code == 200, "Failed to fetch users.csv"

        # Parse CSV content
        df = pd.read_csv(io.StringIO(response.text))

        # Check required columns
        required_columns = ['login', 'cycle_length', 'last_period', 'entries']
        for col in required_columns:
            assert col in df.columns, (
                f"Required column '{col}' missing from users.csv. "
                f"Found columns: {list(df.columns)}"
            )

        # Validate data types if data exists
        if len(df) > 0:
            assert df['login'].dtype == 'object', "Login column should be string type"
            assert pd.api.types.is_numeric_dtype(df['cycle_length']), (
                "cycle_length should be numeric"
            )

    def test_recommendations_json_structure(self):
        """Test that recommendations.json has the expected structure"""
        if not self.github_token:
            pytest.skip("GitHub token not available")

        url = f"{self.base_raw_url}/recommendations.json"
        response = requests.get(url, headers=self.headers)

        assert response.status_code == 200, "Failed to fetch recommendations.json"

        # Parse JSON content
        data = json.loads(response.text)

        # Check required phases
        required_phases = ['menstrual', 'follicular', 'ovulation', 'luteal']
        for phase in required_phases:
            assert phase in data, f"Required phase '{phase}' missing from recommendations"

            # Check required categories within each phase
            required_categories = ['food', 'do', 'dont']
            for category in required_categories:
                assert category in data[phase], (
                    f"Required category '{category}' missing from phase '{phase}'"
                )
                assert isinstance(data[phase][category], list), (
                    f"Category '{category}' in phase '{phase}' should be a list"
                )

    def test_rate_limiting_handling(self):
        """Test that the application handles GitHub API rate limiting"""
        if not self.github_token:
            pytest.skip("GitHub token not available")

        url = f"{self.base_api_url}/users.csv"

        # Mock a rate limit response
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 403
            mock_response.headers = {
                'X-RateLimit-Remaining': '0',
                'X-RateLimit-Reset': '1640995200'
            }
            mock_response.text = '{"message": "API rate limit exceeded"}'
            mock_get.return_value = mock_response

            response = requests.get(url, headers=self.headers)

            assert response.status_code == 403, "Rate limit test should return 403"
            assert 'rate limit' in response.text.lower(), (
                "Rate limit response should mention rate limiting"
            )

    def test_file_update_capability(self):
        """Test that we can update files in the repository (for saving data)"""
        if not self.github_token:
            pytest.skip("GitHub token not available")

        # Test getting file info first (needed for updates)
        url = f"{self.base_api_url}/users.csv"
        response = requests.get(url, headers=self.headers)

        assert response.status_code == 200, "Failed to get file info for update test"

        data = response.json()
        assert 'sha' in data, "File info should include SHA for updates"
        assert isinstance(data['sha'], str), "SHA should be a string"
        assert len(
            data['sha']) == 40, f"SHA should be 40 characters, got: {len(data['sha'])}"

    @pytest.mark.integration
    def test_full_data_loading_workflow(self):
        """Integration test for the complete data loading workflow"""
        if not self.github_token:
            pytest.skip("GitHub token not available")

        # Test users data loading
        users_url = f"{self.base_raw_url}/users.csv"
        users_response = requests.get(users_url, headers=self.headers)
        assert users_response.status_code == 200, "Failed to load users data"

        users_df = pd.read_csv(io.StringIO(users_response.text))
        assert isinstance(
            users_df, pd.DataFrame), "Users data should be a DataFrame"

        # Test recommendations loading
        rec_url = f"{self.base_raw_url}/recommendations.json"
        rec_response = requests.get(rec_url, headers=self.headers)
        assert rec_response.status_code == 200, "Failed to load recommendations data"

        recommendations = json.loads(rec_response.text)
        assert isinstance(
            recommendations, dict), "Recommendations should be a dictionary"

        # Verify the workflow completes without errors
        assert len(users_df.columns) > 0, "Users DataFrame should have columns"
        assert len(recommendations.keys()
                   ) > 0, "Recommendations should have phases"


class TestGitHubConfiguration:
    """Test configuration and setup for GitHub access"""

    def test_repository_exists(self):
        """Test that the repository exists and is accessible"""
        url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}"
        response = requests.get(url)

        # Public info should be accessible even for private repos
        assert response.status_code == 200, (
            f"Repository {REPO_OWNER}/{REPO_NAME} does not exist or is not accessible. "
            f"Status: {response.status_code}"
        )

        repo_info = response.json()
        assert repo_info['name'] == REPO_NAME, (
            f"Repository name mismatch. Expected: {REPO_NAME}, got: {repo_info['name']}"
        )
        assert repo_info['owner']['login'] == REPO_OWNER, (
            f"Repository owner mismatch. Expected: {REPO_OWNER}, got: {repo_info['owner']['login']}"
        )

    def test_repository_is_private(self):
        """Test that the repository is properly configured as private"""
        url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}"
        response = requests.get(url)

        if response.status_code == 200:
            repo_info = response.json()
            assert repo_info['private'] == True, (
                "Repository should be private for secure data storage"
            )


# Utility functions for running tests
def run_quick_test():
    """Quick test function to verify GitHub access is working"""
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("❌ GITHUB_TOKEN not found in environment variables")
        return False

    headers = {"Authorization": f"token {token}"}
    url = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/users.csv"

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            print("✅ GitHub access working!")
            print(f"📄 Users CSV content preview: {response.text[:100]}...")
            return True
        else:
            print(f"❌ GitHub access failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error testing GitHub access: {e}")
        return False


if __name__ == "__main__":
    """Run quick test when script is executed directly"""
    print("🧪 Testing GitHub repository access...")
    print(f"Repository: {REPO_OWNER}/{REPO_NAME}")
    print(f"Branch: {BRANCH}")
    print("-" * 50)

    success = run_quick_test()

    if success:
        print("\n🎉 Basic test passed! Run with pytest for full test suite:")
        print("pytest tests/test_github_access.py -v")
    else:
        print("\n💡 Troubleshooting:")
        print("1. Check your GitHub token: export GITHUB_TOKEN=your_token_here")
        print("2. Verify repository exists and is private")
        print("3. Ensure token has 'repo' scope permissions")
        print("4. Check if files exist in the repository")
