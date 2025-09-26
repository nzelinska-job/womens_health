"""
Menstrual Cycle Tracker - Streamlit Application
Complete cycle tracking with private GitHub repository integration
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import io
import json
import base64
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# Page configuration
st.set_page_config(
    page_title="Menstrual Cycle Tracker",
    page_icon="🌸",
    layout="wide"
)

# ==================== CONFIGURATION CLASSES ====================


class GitHubConfig:
    """GitHub repository configuration with fallback support"""

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
            "app_settings.min_cycle_length", "MIN_CYCLE_LENGTH", "19"))
        self.max_cycle_length = int(self._get_config_value(
            "app_settings.max_cycle_length", "MAX_CYCLE_LENGTH", "60"))
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
MODEL_PICKLE_URL = f"{github_config.base_raw_url}/xgboost.pkl"

# API URLs
USERS_API_URL = f"{github_config.base_api_url}/users.csv"
RECOMMENDATIONS_API_URL = f"{github_config.base_api_url}/recommendations.json"
MODEL_API_URL = f"{github_config.base_api_url}/cycle_model.pkl"

# ==================== CONFIGURATION VALIDATION ====================


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

# ==================== DATA LOADING FUNCTIONS ====================


@st.cache_data
def load_users_data():
    """Load user data from private GitHub repository"""
    try:
        headers = github_config.headers

        # Try raw URL first (faster)
        response = requests.get(USERS_CSV_URL, headers=headers)
        if response.status_code == 200:
            return pd.read_csv(io.StringIO(response.text))

        # If raw URL fails, try API
        response = requests.get(USERS_API_URL, headers=headers)
        if response.status_code == 200:
            content = response.json()
            file_content = base64.b64decode(content['content']).decode('utf-8')
            return pd.read_csv(io.StringIO(file_content))
        else:
            if app_config.debug:
                st.warning(
                    f"⚠️ Could not load users data. Status: {response.status_code}")
            return pd.DataFrame(columns=['login', 'cycle_length', 'last_period', 'entries'])
    except Exception as e:
        if app_config.debug:
            st.error(f"❌ Error loading users data: {e}")
        return pd.DataFrame(columns=['login', 'cycle_length', 'last_period', 'entries'])


@st.cache_data
def load_recommendations():
    """Load recommendations from private GitHub repository"""
    try:
        headers = github_config.headers

        # Try raw URL first
        response = requests.get(RECOMMENDATIONS_JSON_URL, headers=headers)
        if response.status_code == 200:
            return json.loads(response.text)

        # Try API if raw URL fails
        response = requests.get(RECOMMENDATIONS_API_URL, headers=headers)
        if response.status_code == 200:
            content = response.json()
            file_content = base64.b64decode(content['content']).decode('utf-8')
            return json.loads(file_content)
        else:
            if app_config.debug:
                st.warning(
                    "⚠️ Could not load recommendations. Using defaults.")
            return get_default_recommendations()
    except Exception as e:
        if app_config.debug:
            st.error(f"❌ Error loading recommendations: {e}")
        return get_default_recommendations()


@st.cache_resource
def load_model():
    """Load machine learning model from private GitHub repository"""
    try:
        import xgboost

        headers = github_config.headers

        # Try raw URL first
        response = requests.get(MODEL_PICKLE_URL, headers=headers)
        if response.status_code == 200:
            model_data = pickle.loads(response.content)

            # Check what was loading
            if app_config.debug:
                st.write(f"**Model type:** {type(model_data)}")
                if isinstance(model_data, dict):
                    st.write(f"**Dict keys:** {list(model_data.keys())}")

            # Check if it's a model
            if hasattr(model_data, 'predict'):
                return model_data
            elif isinstance(model_data, dict) and 'model' in model_data:
                # May be a dict with model and metadata
                return model_data['model']
            else:
                if app_config.debug:
                    st.warning("⚠️ Loaded file is not a trained ML model")
                return None

        return None
    except Exception as e:
        if app_config.debug:
            st.error(f"ML model loading error: {e}")
        return None

# ==================== DATA SAVING FUNCTIONS ====================


def save_user_data_to_github(login, data):
    """Save user data back to private GitHub repository"""
    try:
        headers = github_config.headers

        # First, get current file content and SHA
        response = requests.get(USERS_API_URL, headers=headers)
        if response.status_code == 200:
            current_data = response.json()
            current_sha = current_data['sha']

            # Load current users
            current_content = base64.b64decode(
                current_data['content']).decode('utf-8')
            df = pd.read_csv(io.StringIO(current_content))

            # Update or add user data
            user_exists = df['login'] == login
            if user_exists.any():
                # Update existing user
                for key, value in data.items():
                    if key != 'login':
                        if key == 'entries':
                            # Handle entries list properly
                            df.loc[user_exists, key] = json.dumps(
                                value) if isinstance(value, list) else value
                        else:
                            df.loc[user_exists, key] = value
            else:
                # Add new user
                new_row = pd.DataFrame([data])
                df = pd.concat([df, new_row], ignore_index=True)

            # Convert back to CSV
            csv_content = df.to_csv(index=False)
            encoded_content = base64.b64encode(csv_content.encode()).decode()

            # Update file on GitHub
            update_data = {
                "message": f"Update user data for {login} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "content": encoded_content,
                "sha": current_sha
            }

            update_response = requests.put(USERS_API_URL,
                                           json=update_data,
                                           headers=headers)

            if update_response.status_code == 200:
                st.success("✅ Data successfully saved to repository!")
                # Clear cache to reload fresh data
                st.cache_data.clear()
                return True
            else:
                st.error(f"❌ Error saving data: {update_response.status_code}")
                if app_config.debug:
                    st.error(f"Response: {update_response.text}")
                return False
        else:
            st.error("❌ Could not access users file in repository")
            return False

    except Exception as e:
        st.error(f"❌ Exception saving data: {e}")
        return False


def save_user_data(login, data):
    """Wrapper function for saving user data"""
    return save_user_data_to_github(login, data)

# ==================== DEFAULT DATA ====================


def get_default_recommendations():
    """Default recommendations when file is not available"""
    return {
        "menstrual": {
            "food": ["🥛 Dairy products", "🥬 Leafy greens", "🍫 Dark chocolate", "🫐 Berries", "🐟 Iron-rich fish", "🥩 Lean red meat"],
            "do": ["🛁 Warm baths", "🧘‍♀️ Meditation", "😴 Adequate sleep", "🚶‍♀️ Light walks", "📚 Reading", "🎵 Relaxing music"],
            "dont": ["🏃‍♀️ Intense workouts", "☕ Too much caffeine", "🧂 Salty foods", "📱 Excessive stress", "🍷 Alcohol", "🥶 Cold foods"]
        },
        "follicular": {
            "food": ["🥗 Fresh vegetables", "🍎 Fruits", "🐟 Fish", "🌰 Nuts", "🥑 Healthy fats", "🍓 Antioxidant-rich berries"],
            "do": ["💪 Strength training", "🎯 Goal planning", "📚 Learning new skills", "🎨 Creative projects", "👥 Social activities", "🌅 Morning workouts"],
            "dont": ["🍟 Processed foods", "🍷 Excess alcohol", "😰 Overworking", "🌙 Late bedtime", "🍰 Sugary snacks", "📱 Digital overwhelm"]
        },
        "ovulation": {
            "food": ["🥑 Avocado", "🍓 Strawberries", "🐣 Eggs", "🥜 Almonds", "🐟 Omega-3 rich fish", "🥬 Spinach"],
            "do": ["🏃‍♀️ Cardio workouts", "👥 Socializing", "💼 Important meetings", "💃 Dancing", "🗣️ Public speaking", "💪 High-intensity exercise"],
            "dont": ["🍕 Heavy meals", "😴 Sleep deprivation", "🚫 Social isolation", "📵 Ignoring your energy", "🥤 Sugary drinks", "😷 Skipping self-care"]
        },
        "luteal": {
            "food": ["🍠 Sweet potato", "🥒 Cucumbers", "🍵 Herbal tea", "🥥 Coconut", "🥜 Magnesium-rich nuts", "🍌 Bananas"],
            "do": ["🧘‍♀️ Yoga", "📝 Journaling", "🛀 Relaxation", "🎵 Calming music", "🏠 Nesting activities", "💤 Extra sleep"],
            "dont": ["🍰 Refined sugars", "📊 Stressful projects", "💤 Sleep disruption", "🔥 Conflicts", "☕ Late-day caffeine", "🏃‍♀️ Overexertion"]
        }
    }

# ==================== CALCULATION FUNCTIONS ====================


def calculate_cycle_day(last_period_date):
    """Calculate current cycle day based on last period date"""
    today = datetime.now().date()
    last_period = datetime.strptime(last_period_date, '%Y-%m-%d').date()
    return (today - last_period).days + 1


def determine_cycle_phase(cycle_day, cycle_length):
    """Determine current cycle phase based on cycle day and length"""
    if cycle_day <= 5:
        return "menstrual"
    elif cycle_day <= cycle_length // 2:
        return "follicular"
    elif cycle_day <= (cycle_length // 2) + 2:
        return "ovulation"
    else:
        return "luteal"


def predict_next_period(last_period_date, cycle_length):
    """Predict next period date"""
    last_period = datetime.strptime(last_period_date, '%Y-%m-%d').date()
    return last_period + timedelta(days=cycle_length)

# ==================== VISUALIZATION FUNCTIONS ====================


def create_cycle_chart(cycle_day, cycle_length, phase):
    """Create interactive cycle visualization chart"""
    phases = []
    colors = []

    # Menstrual phase (days 1-5)
    phases.extend(['Menstrual'] * 5)
    colors.extend(['#FF6B6B'] * 5)

    # Follicular phase
    follicular_days = max(1, (cycle_length // 2) - 5)
    phases.extend(['Follicular'] * follicular_days)
    colors.extend(['#4ECDC4'] * follicular_days)

    # Ovulation phase
    phases.extend(['Ovulation'] * 3)
    colors.extend(['#45B7D1'] * 3)

    # Luteal phase
    luteal_days = cycle_length - len(phases)
    if luteal_days > 0:
        phases.extend(['Luteal'] * luteal_days)
        colors.extend(['#96CEB4'] * luteal_days)

    # Create pie chart
    fig = go.Figure(data=go.Pie(
        labels=phases,
        values=[1] * len(phases),
        marker_colors=colors,
        hole=0.4,
        textinfo='none',
        hovertemplate='<b>%{label}</b><br>Day: %{pointNumber}<extra></extra>'
    ))

    # Add current day marker
    if cycle_day <= len(phases):
        angle = (cycle_day - 1) * 360 / len(phases)
        fig.add_annotation(
            x=0.5 + 0.3 * np.cos(np.radians(angle - 90)),
            y=0.5 + 0.3 * np.sin(np.radians(angle - 90)),
            text=f"Day {cycle_day}",
            showarrow=True,
            arrowhead=2,
            arrowcolor="red",
            arrowwidth=3,
            font=dict(size=14, color="red")
        )

    fig.update_layout(
        title=f"Your Menstrual Cycle - Current Phase: {phase.title()}",
        showlegend=True,
        font=dict(size=12),
        height=500,
        margin=dict(t=60, b=40, l=40, r=40)
    )

    return fig


def predict_cycle_phase_ml(model, features):
    """Use ML model to predict cycle phase"""
    if model is not None:
        try:
            prediction = model.predict([features])
            return prediction[0]
        except Exception as e:
            if app_config.debug:
                st.error(f"ML prediction error: {e}")
            return None
    return None

# ==================== MAIN APPLICATION ====================


def main():
    """Main application function"""
    st.title("🌸 Menstrual Cycle Tracker")
    st.markdown(
        "*Track your cycle, understand your body, optimize your wellbeing*")

    # Configuration validation
    is_valid, errors = validate_configuration()
    if not is_valid:
        st.error("⚠️ **Configuration Issues Detected:**")
        for error in errors:
            st.error(f"• {error}")

        with st.expander("🔧 How to fix configuration"):
            st.markdown("""
            **Option 1: Environment Variables (.env file)**
            ```bash
            GITHUB_TOKEN=ghp_your_token_here
            GITHUB_OWNER=nzelinska-job
            GITHUB_REPO=cycle-tracker-data
            ```
            
            **Option 2: Streamlit Secrets (.streamlit/secrets.toml)**
            ```toml
            github_token = "ghp_your_token_here"
            
            [data_sources]
            repo_owner = "nzelinska-job"
            repo_name = "cycle-tracker-data"
            ```
            
            **Get GitHub Token:**
            1. Go to GitHub → Settings → Developer settings → Personal access tokens
            2. Generate new token with `repo` scope
            3. Copy and paste in your configuration
            """)
        st.stop()

    # Debug information
    if app_config.debug:
        with st.expander("🐛 Debug Information"):
            st.write(
                f"**Repository:** {github_config.owner}/{github_config.repo}")
            st.write(f"**Environment:** {app_config.environment}")
            st.write(
                f"**Token configured:** {'✅' if github_config.token else '❌'}")
            st.write(f"**URLs:**")
            st.write(f"- Users CSV: {USERS_CSV_URL}")
            st.write(f"- Recommendations: {RECOMMENDATIONS_JSON_URL}")

    st.markdown("---")

    # Initialize session state variables
    if 'user_login' not in st.session_state:
        st.session_state.user_login = None
    if 'user_data' not in st.session_state:
        st.session_state.user_data = None
    if 'show_recommendations' not in st.session_state:
        st.session_state.show_recommendations = False
    if 'daily_entries' not in st.session_state:
        st.session_state.daily_entries = []

    # New session state variables for registration
    if 'registration_mode' not in st.session_state:
        st.session_state.registration_mode = False
    if 'new_user_login' not in st.session_state:
        st.session_state.new_user_login = ""
    if 'show_registration_form' not in st.session_state:
        st.session_state.show_registration_form = False

    # Load data from private GitHub repository
    users_df = load_users_data()
    recommendations = load_recommendations()
    model = load_model()

    # Login or user selection section
    if st.session_state.user_login is None:
        st.subheader("👋 Welcome! Please sign in")
        st.markdown(
            "Enter your unique login to access your cycle data or create a new profile.")

        # Show login input only if not in registration mode
        if not st.session_state.registration_mode:
            login = st.text_input(
                "🆔 Enter your unique login:", placeholder="your_username")

            if st.button("🔍 Check Login", type="primary"):
                if login:
                    # Check if user exists in database
                    existing_user = users_df[users_df['login'] == login]

                    if not existing_user.empty:
                        # Existing user found
                        st.session_state.user_login = login
                        user_dict = existing_user.iloc[0].to_dict()
                        # Handle entries field properly
                        if 'entries' in user_dict and isinstance(user_dict['entries'], str):
                            try:
                                user_dict['entries'] = json.loads(
                                    user_dict['entries'])
                            except:
                                user_dict['entries'] = []
                        st.session_state.user_data = user_dict
                        st.success(
                            f"Welcome back, {login}! Loading your data...")
                        st.rerun()
                    else:
                        # New user - switch to registration mode
                        st.session_state.registration_mode = True
                        st.session_state.new_user_login = login
                        st.session_state.show_registration_form = True
                        st.info(
                            "👤 New user detected! Please complete your profile setup.")
                        st.rerun()
                else:
                    st.warning("⚠️ Please enter a login name!")

        # Show registration form if in registration mode
        if st.session_state.registration_mode and st.session_state.show_registration_form:
            st.markdown("---")
            st.subheader(
                f"📝 Create Profile for: **{st.session_state.new_user_login}**")

            # Back button to return to login
            if st.button("← Back to Login", type="secondary"):
                st.session_state.registration_mode = False
                st.session_state.new_user_login = ""
                st.session_state.show_registration_form = False
                st.rerun()

            # Registration form using st.form to prevent resets
            with st.form("registration_form"):
                st.markdown("**Complete your cycle information:**")

                col1, col2 = st.columns(2)
                with col1:
                    cycle_length = st.number_input(
                        "📊 Typical cycle length (days):",
                        min_value=app_config.min_cycle_length,
                        max_value=app_config.max_cycle_length,
                        value=app_config.default_cycle_length,
                        help=f"Most cycles are between {app_config.min_cycle_length}-{app_config.max_cycle_length} days"
                    )

                with col2:
                    last_period = st.date_input(
                        "📅 Date of last menstruation:",
                        max_value=datetime.now().date(),
                        help="Select the first day of your last period"
                    )

                # Form submit button
                submitted = st.form_submit_button(
                    "💾 Create Profile", type="primary")

                if submitted:
                    # Validate inputs
                    if cycle_length and last_period:
                        # Create new user profile
                        new_user_data = {
                            'login': st.session_state.new_user_login,
                            'cycle_length': cycle_length,
                            'last_period': last_period.strftime('%Y-%m-%d'),
                            'entries': []
                        }

                        # Save to GitHub
                        if save_user_data(st.session_state.new_user_login, new_user_data):
                            # Successfully created profile
                            st.session_state.user_login = st.session_state.new_user_login
                            st.session_state.user_data = new_user_data

                            # Reset registration state
                            st.session_state.registration_mode = False
                            st.session_state.new_user_login = ""
                            st.session_state.show_registration_form = False

                            st.success(
                                "✅ Profile created successfully! Welcome to your cycle tracker!")
                            st.rerun()
                        else:
                            st.error(
                                "❌ Failed to save profile. Please try again.")
                    else:
                        st.error("⚠️ Please fill in all required fields.")

    else:
        # User is logged in - main application interface
        user_data = st.session_state.user_data

        # Calculate current cycle status
        cycle_day = calculate_cycle_day(user_data['last_period'])
        cycle_length = user_data['cycle_length']
        current_phase = determine_cycle_phase(cycle_day, cycle_length)
        next_period = predict_next_period(
            user_data['last_period'], cycle_length)
        days_to_period = (next_period - datetime.now().date()).days

        # Header with user information
        st.subheader(f"Hello, {st.session_state.user_login}! 👋")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Cycle Day", f"{cycle_day}/{cycle_length}")
        with col2:
            phase_names = {
                'menstrual': '🔴 Menstrual',
                'follicular': '🌱 Follicular',
                'ovulation': '⭐ Ovulation',
                'luteal': '🍂 Luteal'
            }
            st.metric("🌟 Current Phase", phase_names.get(
                current_phase, current_phase))
        with col3:
            if days_to_period > 0:
                st.metric("📅 Next Period", f"in {days_to_period} days")
            else:
                st.metric("📅 Period", "Due now" if days_to_period ==
                          0 else f"{abs(days_to_period)} days overdue")
        with col4:
            st.metric("📈 Cycle Progress",
                      f"{int((cycle_day/cycle_length)*100)}%")

        st.markdown("---")

        # Daily data input form
        st.subheader("📝 Today's Data Entry")
        st.markdown(
            "Track your daily symptoms and wellbeing to get personalized insights.")

        col1, col2, col3 = st.columns(3)
        with col1:
            stress_level = st.slider(
                "😰 Stress Level",
                -3, -1, 0,
                help="Rate your stress from -3 (very stressed) to 0 (very calm)"
            )
        with col2:
            mood = st.selectbox(
                "😊 Overall Mood",
                ["😢😢 depressed", "😢 Sad", "😐 Neutral", "😊 Good", "😍 Excellent"],
                index=2,
                help="How are you feeling today overall?"
            )
        with col3:
            energy_level = st.slider(
                "⚡ Energy Level",
                -2, 0, 2,
                help="Rate your energy from -2 (very low) to 2 (very high)"
            )

        # Symptoms selection
        st.markdown("**🩺 Symptoms (select all that apply):**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("*Physical Symptoms*")
            symptoms_physical = st.multiselect(
                "Physical",
                ["Headache", "Breast tenderness", "Sleep issues",
                 "Cramps", "Bloating", "Acne", "Back pain", "Fatigue"],
                help="Select any physical symptoms you're experiencing",
                label_visibility="collapsed"
            )
        with col2:
            st.markdown("*Emotional Symptoms*")
            symptoms_emotional = st.multiselect(
                "Emotional",
                ["Irritability", "Anxiety", "Mood swings", "Depression",
                 "Food cravings", "Fatigue", "Difficulty concentrating"],
                help="Select any emotional symptoms you're experiencing",
                label_visibility="collapsed"
            )

        # Combine all symptoms
        all_symptoms = symptoms_physical + symptoms_emotional

        # Additional notes
        additional_notes = st.text_area(
            "📋 Additional Notes",
            placeholder="Any other observations about your day, body, or mood...",
            help="Optional: Add any other relevant information"
        )

        # Save daily data button
        if st.button("💾 Save Today's Data", type="primary"):
            today = datetime.now().date().strftime('%Y-%m-%d')

            daily_entry = {
                'date': today,
                'cycle_day': cycle_day,
                'stress_level': stress_level,
                'mood': mood,
                'energy_level': energy_level,
                'symptoms': all_symptoms,
                'notes': additional_notes
            }

            # Add entry to user data
            entries = user_data.get('entries', [])
            # Remove entry for today if it exists
            entries = [e for e in entries if e.get('date') != today]
            # Add new entry
            entries.append(daily_entry)
            user_data['entries'] = entries

            # Save to GitHub
            if save_user_data(st.session_state.user_login, user_data):
                st.session_state.user_data = user_data

        st.markdown("---")

        # Cycle information and chart section
        if st.button("📊 Show Cycle Information", type="secondary"):
            st.subheader("📈 Your Menstrual Cycle Overview")

            # Create and display cycle chart
            cycle_chart = create_cycle_chart(
                cycle_day, cycle_length, current_phase)
            st.plotly_chart(cycle_chart, use_container_width=True)

            # Phase information
            phase_descriptions = {
                'menstrual': "🔴 **Menstrual Phase** (Days 1-5) - Time for rest and renewal. Hormone levels are low. Focus on self-care, gentle activities, and iron-rich foods.",
                'follicular': "🌱 **Follicular Phase** (Days 6-14) - Time of growth and new energy. Estrogen is rising. Great time for new projects, physical activity, and social connections.",
                'ovulation': "⭐ **Ovulation Phase** (Days 14-16) - Peak fertility and energy. Highest estrogen levels. Perfect time for important meetings, presentations, and high-intensity activities.",
                'luteal': "🍂 **Luteal Phase** (Days 17-28) - Time for preparation and reflection. Progesterone dominates. Focus on completing tasks, self-reflection, and comfort foods."
            }

            st.info(phase_descriptions.get(current_phase, "Unknown phase"))

            # ML model prediction (if available)
            if model is not None:
                st.subheader("🤖 AI Phase Prediction")
                features = [cycle_day, cycle_length, stress_level,
                            energy_level, len(symptoms_physical)]
                predicted_phase = predict_cycle_phase_ml(model, features)
                if predicted_phase:
                    st.write(f"**AI Predicted Phase:** {predicted_phase}")
                    if predicted_phase != current_phase:
                        st.warning(
                            f"⚠️ AI prediction differs from calculated phase. This might indicate cycle irregularity.")

            # Enable recommendations button
            st.session_state.show_recommendations = True

        # Recommendations section
        if st.session_state.show_recommendations:
            if st.button("💡 Get Phase-Based Recommendations", type="secondary"):
                st.subheader("🎯 Personalized Recommendations")
                st.markdown(
                    f"*Based on your current **{current_phase}** phase*")

                current_recommendations = recommendations.get(
                    current_phase, {})

                # Create three tabs for different recommendation types
                tab1, tab2, tab3 = st.tabs(
                    ["🍽️ Food", "✅ Do This", "❌ Avoid This"])

                with tab1:
                    st.markdown("### 🍽️ Recommended Foods")
                    st.markdown(
                        "*Foods that support your body during this phase*")

                    food_items = current_recommendations.get('food', [])
                    if food_items:
                        cols = st.columns(2)
                        for i, item in enumerate(food_items):
                            with cols[i % 2]:
                                st.write(f"• {item}")
                    else:
                        st.write("No specific food recommendations available.")

                with tab2:
                    st.markdown("### ✅ Recommended Activities")
                    st.markdown(
                        "*Activities that align with your current energy and hormones*")

                    do_items = current_recommendations.get('do', [])
                    if do_items:
                        cols = st.columns(2)
                        for i, item in enumerate(do_items):
                            with cols[i % 2]:
                                st.write(f"• {item}")
                    else:
                        st.write(
                            "No specific activity recommendations available.")

                with tab3:
                    st.markdown("### ❌ Things to Avoid")
                    st.markdown(
                        "*What might not serve you well during this phase*")

                    dont_items = current_recommendations.get('dont', [])
                    if dont_items:
                        cols = st.columns(2)
                        for i, item in enumerate(dont_items):
                            with cols[i % 2]:
                                st.write(f"• {item}")
                    else:
                        st.write("No specific restrictions for this phase.")

        # Analytics section (if user has entries)
        if user_data.get('entries'):
            st.markdown("---")
            if st.button("📈 View Analytics", type="secondary"):
                st.subheader("📊 Your Cycle Analytics")

                entries = user_data['entries']
                if len(entries) > 1:
                    df_entries = pd.DataFrame(entries)
                    df_entries['date'] = pd.to_datetime(df_entries['date'])

                    # Stress and energy trends
                    col1, col2 = st.columns(2)

                    with col1:
                        fig_stress = px.line(df_entries, x='date', y='stress_level',
                                             title='Stress Level Over Time',
                                             line_shape='spline')
                        fig_stress.update_traces(line_color='#FF6B6B')
                        st.plotly_chart(fig_stress, use_container_width=True)

                    with col2:
                        fig_energy = px.line(df_entries, x='date', y='energy_level',
                                             title='Energy Level Over Time',
                                             line_shape='spline')
                        fig_energy.update_traces(line_color='#45B7D1')
                        st.plotly_chart(fig_energy, use_container_width=True)

                    # Symptom frequency
                    all_symptoms_flat = []
                    for entry in entries:
                        all_symptoms_flat.extend(entry.get('symptoms', []))

                    if all_symptoms_flat:
                        symptom_counts = pd.Series(
                            all_symptoms_flat).value_counts()
                        fig_symptoms = px.bar(x=symptom_counts.values, y=symptom_counts.index,
                                              orientation='h',
                                              title='Most Common Symptoms',
                                              color=symptom_counts.values,
                                              color_continuous_scale='Viridis')
                        st.plotly_chart(fig_symptoms, use_container_width=True)
                else:
                    st.info(
                        "📈 Analytics will be available after you log more daily entries.")

        # Logout section
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 1, 1])

        with col2:
            if st.button("🔄 Update Profile", type="secondary"):
                st.session_state.show_profile_update = True

        with col3:
            if st.button("🚪 Logout", type="secondary"):
                # Clear session state
                st.session_state.user_login = None
                st.session_state.user_data = None
                st.session_state.show_recommendations = False
                st.session_state.daily_entries = []
                st.success("Logged out successfully!")
                st.rerun()

        # Profile update section
        if st.session_state.get('show_profile_update', False):
            st.markdown("---")
            st.subheader("🔄 Update Profile")

            col1, col2 = st.columns(2)
            with col1:
                new_cycle_length = st.number_input(
                    "📊 Update cycle length:",
                    min_value=app_config.min_cycle_length,
                    max_value=app_config.max_cycle_length,
                    value=user_data['cycle_length']
                )
            with col2:
                new_last_period = st.date_input(
                    "📅 Update last period date:",
                    value=datetime.strptime(
                        user_data['last_period'], '%Y-%m-%d').date(),
                    max_value=datetime.now().date()
                )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save Changes", type="primary"):
                    user_data['cycle_length'] = new_cycle_length
                    user_data['last_period'] = new_last_period.strftime(
                        '%Y-%m-%d')

                    if save_user_data(st.session_state.user_login, user_data):
                        st.session_state.user_data = user_data
                        st.session_state.show_profile_update = False
                        st.success("Profile updated successfully!")
                        st.rerun()

            with col2:
                if st.button("❌ Cancel"):
                    st.session_state.show_profile_update = False
                    st.rerun()

        # Footer with helpful information
        st.markdown("---")
        with st.expander("ℹ️ About This App"):
            st.markdown("""
            **Menstrual Cycle Tracker** helps you understand your body's natural rhythms and optimize your wellbeing.
            
            **Features:**
            - 📝 Track daily symptoms, mood, and energy levels
            - 📊 Visualize your cycle phases with interactive charts  
            - 💡 Get personalized recommendations for each phase
            - 📈 View analytics and trends over time
            - 🤖 AI-powered phase prediction (when model is available)
            - 🔒 Secure data storage in private GitHub repository
            
            **Phase Information:**
            - **Menstrual (Days 1-5):** Rest, renewal, gentle self-care
            - **Follicular (Days 6-14):** Growth, energy, new projects
            - **Ovulation (Days 14-16):** Peak energy, social activities  
            - **Luteal (Days 17-28):** Reflection, completion, comfort
            
            **Data Privacy:** 
            - Your data is stored securely in a private GitHub repository
            - Only you have access to your personal information
            - Data is encrypted during transmission
            
            **Disclaimer:** 
            This app is for informational and educational purposes only. It should not replace professional medical advice, diagnosis, or treatment. Always consult with a healthcare provider for medical concerns.
            
            **Version:** 1.0 | **Last Updated:** {datetime.now().strftime('%Y-%m-%d')}
            """)

        # Emergency contact info
        with st.expander("🚨 When to Seek Medical Help"):
            st.markdown("""
            **Consult a healthcare provider if you experience:**
            - Periods that last longer than 7 days
            - Cycles shorter than 21 days or longer than 35 days
            - Severe pain that interferes with daily activities
            - Heavy bleeding (changing pad/tampon every hour)
            - Bleeding between periods
            - No period for 3+ months (if not pregnant/breastfeeding)
            - Sudden changes in your cycle pattern
            
            **Emergency signs - seek immediate medical attention:**
            - Severe abdominal pain
            - Heavy bleeding with large clots
            - Signs of infection (fever, unusual discharge, strong odor)
            - Fainting or severe dizziness
            """)


if __name__ == "__main__":
    main()
