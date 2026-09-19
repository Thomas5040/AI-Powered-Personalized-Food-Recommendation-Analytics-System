import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from datetime import datetime
import hashlib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Personalized Food Recommendation System",
    page_icon="🍱",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "Final_Food_Recommendation_Dataset.xlsx"
KMEANS_FILE = BASE_DIR / "food_kmeans_model.pkl"
SCALER_FILE = BASE_DIR / "food_scaler.pkl"
RECOMMENDATION_SCALER_FILE = BASE_DIR / "recommendation_scaler.pkl"

HISTORY_FILE = BASE_DIR / "Client_Recommendation_History.xlsx"


# ============================================================
# ADMIN LOGIN DETAILS
# ============================================================

ADMIN_USERNAME = "Thomas"
ADMIN_PASSWORD = "Tom@12345"


# ============================================================
# SESSION STATE
# ============================================================

if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

if "page" not in st.session_state:
    st.session_state.page = "recommendation"


# ============================================================
# LOAD FOOD DATA
# ============================================================

@st.cache_data
def load_food_data():

    return pd.read_excel(DATA_FILE)


# ============================================================
# LOAD ML MODELS
# ============================================================

@st.cache_resource
def load_models():

    kmeans = joblib.load(KMEANS_FILE)

    scaler = joblib.load(SCALER_FILE)

    recommendation_scaler = joblib.load(
        RECOMMENDATION_SCALER_FILE
    )

    return kmeans, scaler, recommendation_scaler


# ============================================================
# LOAD PROJECT
# ============================================================

try:

    df = load_food_data()

    kmeans, scaler, recommendation_scaler = load_models()

except Exception as e:

    st.error("Error loading project files.")

    st.error(str(e))

    st.info(
        "Make sure the Excel file and all three .pkl files "
        "are inside the same folder as app.py."
    )

    st.stop()


# ============================================================
# HISTORY FILE
# ============================================================

def create_history_file():

    if not HISTORY_FILE.exists():

        history_columns = [

            "Client_ID",
            "Client_Name",
            "Age",
            "Phone_Number",
            "Address",
            "Gender",
            "Dietary_Type",
            "Health_Goal",
            "High_Protein",
            "High_Fiber",
            "Low_Sugar",
            "Low_Sodium",
            "Low_Saturated_Fat",
            "Recommendation_Type",
            "Recommendation_Date",
            "Recommended_Food",
            "Food_Category",
            "Food_Profile",
            "Calories",
            "Protein_g",
            "Fiber_g",
            "Sugar_g",
            "Sodium_mg",
            "Recommendation_Score"

        ]

        history_df = pd.DataFrame(
            columns=history_columns
        )

        history_df.to_excel(
            HISTORY_FILE,
            index=False
        )


create_history_file()


# ============================================================
# LOAD HISTORY
# ============================================================

def load_history():

    if HISTORY_FILE.exists():

        try:

            return pd.read_excel(HISTORY_FILE)

        except Exception:

            return pd.DataFrame()

    return pd.DataFrame()


# ============================================================
# SAVE HISTORY
# ============================================================

def save_history(new_records):

    history = load_history()

    updated_history = pd.concat(
        [history, new_records],
        ignore_index=True
    )

    updated_history.to_excel(
        HISTORY_FILE,
        index=False
    )


# ============================================================
# GENERATE CLIENT ID
# ============================================================

def generate_client_id(phone_number):

    phone_digits = "".join(
        filter(str.isdigit, str(phone_number))
    )

    if len(phone_digits) >= 6:

        return "CL-" + phone_digits[-6:]

    hashed = hashlib.md5(
        str(phone_number).encode()
    ).hexdigest()[:6]

    return "CL-" + hashed


# ============================================================
# RECOMMENDATION FUNCTION
# ============================================================

def get_recommendations(
    data,
    high_protein=False,
    high_fiber=False,
    low_sugar=False,
    low_sodium=False,
    low_saturated_fat=False,
    dietary_type="All",
    category="All",
    excluded_foods=None,
    top_n=10
):

    result = data.copy()

    if excluded_foods is None:

        excluded_foods = []

    # Dietary filter

    if dietary_type != "All":

        result = result[
            result["Dietary_Type"] == dietary_type
        ]

    # Category filter

    if category != "All":

        result = result[
            result["Category"] == category
        ]

    # Exclude previous foods

    if excluded_foods:

        result = result[
            ~result["Food_Name"].isin(excluded_foods)
        ]

    # Score columns

    score_columns = []

    if high_protein:

        score_columns.append(
            "Protein_Score"
        )

    if high_fiber:

        score_columns.append(
            "Fiber_Score"
        )

    if low_sugar:

        score_columns.append(
            "Low_Sugar_Score"
        )

    if low_sodium:

        score_columns.append(
            "Low_Sodium_Score"
        )

    if low_saturated_fat:

        score_columns.append(
            "Low_Saturated_Fat_Score"
        )

    # Personalized score

    if len(score_columns) == 0:

        result["Personalized_Score"] = (
            result["Recommendation_Score"]
        )

    else:

        result["Personalized_Score"] = (
            result[score_columns]
            .mean(axis=1)
            * 100
        )

    # Sort

    result = result.sort_values(
        "Personalized_Score",
        ascending=False
    )

    return result.head(top_n)


# ============================================================
# ADMIN LOGIN
# ============================================================

def admin_login():

    st.markdown(
        """
        <style>

        .admin-box {
            background-color: #f7f7f7;
            padding: 25px;
            border-radius: 12px;
            border: 1px solid #dddddd;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🔐 Admin Login")

    st.write(
        "Enter administrator credentials to access "
        "the dashboard."
    )

    col1, col2 = st.columns([1, 1])

    with col1:

        username = st.text_input(
            "Username"
        )

        password = st.text_input(
            "Password",
            type="password"
        )

        login = st.button(
            "🔓 Login",
            type="primary",
            use_container_width=True
        )

        if login:

            if (
                username == ADMIN_USERNAME
                and password == ADMIN_PASSWORD
            ):

                st.session_state.admin_logged_in = True

                st.session_state.page = "admin"

                st.success(
                    "Login successful."
                )

                st.rerun()

            else:

                st.error(
                    "Invalid username or password."
                )

    with col2:

        st.info(
            """
            ### Administrator Access

            The admin dashboard provides:

            • Client analytics

            • Recommendation history

            • Food dataset management

            • KPI dashboard

            • Charts

            • AI dataset assistant

            • Excel downloads
            """
        )


# ============================================================
# TOP NAVIGATION
# ============================================================

def navigation():

    col1, col2, col3 = st.columns(
        [7, 2, 2]
    )

    with col1:

        st.markdown(
            "## 🍱 Personalized Food Recommendation System"
        )

    with col2:

        if st.session_state.admin_logged_in:

            if st.button(
                "📊 Admin Dashboard",
                use_container_width=True
            ):

                st.session_state.page = "admin"

                st.rerun()

        else:

            if st.button(
                "🔐 Admin Login",
                use_container_width=True
            ):

                st.session_state.page = "login"

                st.rerun()

    with col3:

        if st.session_state.admin_logged_in:

            if st.button(
                "🚪 Logout",
                use_container_width=True
            ):

                st.session_state.admin_logged_in = False

                st.session_state.page = "recommendation"

                st.rerun()

        else:

            if st.button(
                "🏠 Home",
                use_container_width=True
            ):

                st.session_state.page = "recommendation"

                st.rerun()

    st.divider()


# ============================================================
# ADMIN DASHBOARD
# ============================================================

def admin_dashboard():

    st.title("📊 Admin Dashboard")

    st.caption(
        "Overall business intelligence and food recommendation analytics"
    )

    history = load_history()

    # ========================================================
    # KPI SECTION
    # ========================================================

    st.subheader("📌 Overall KPIs")

    total_foods = len(df)

    total_clients = (
        history["Client_ID"].nunique()
        if not history.empty
        else 0
    )

    total_recommendations = (
        len(history)
        if not history.empty
        else 0
    )

    vegetarian_count = (
        df["Dietary_Type"]
        .eq("Vegetarian")
        .sum()
    )

    nonveg_count = (
        df["Dietary_Type"]
        .eq("Non-Vegetarian")
        .sum()
    )

    unique_recommended_foods = (
        history["Recommended_Food"].nunique()
        if not history.empty
        else 0
    )

    today_count = 0

    if not history.empty:

        dates = pd.to_datetime(
            history["Recommendation_Date"],
            errors="coerce"
        )

        today = datetime.now().date()

        today_count = (
            dates.dt.date == today
        ).sum()

    k1, k2, k3, k4, k5, k6 = st.columns(6)

    with k1:

        st.metric(
            "👥 Total Clients",
            total_clients
        )

    with k2:

        st.metric(
            "🍽️ Recommendations",
            total_recommendations
        )

    with k3:

        st.metric(
            "🥗 Vegetarian Foods",
            vegetarian_count
        )

    with k4:

        st.metric(
            "🍗 Non-Veg Foods",
            nonveg_count
        )

    with k5:

        st.metric(
            "🍱 Foods Recommended",
            unique_recommended_foods
        )

    with k6:

        st.metric(
            "📅 Today",
            today_count
        )

    st.divider()

    # ========================================================
    # CHARTS
    # ========================================================

    if not history.empty:

        st.subheader(
            "📈 Recommendation Analytics"
        )

        chart_col1, chart_col2 = st.columns(2)

        # ----------------------------------------------------
        # Dietary type chart
        # ----------------------------------------------------

        with chart_col1:

            st.write(
                "### 🥗 Recommendations by Dietary Type"
            )

            dietary_chart = (
                history[
                    "Dietary_Type"
                ]
                .value_counts()
            )

            st.bar_chart(
                dietary_chart
            )

        # ----------------------------------------------------
        # Health goal chart
        # ----------------------------------------------------

        with chart_col2:

            st.write(
                "### 🎯 Recommendations by Health Goal"
            )

            goal_chart = (
                history[
                    "Health_Goal"
                ]
                .value_counts()
            )

            st.bar_chart(
                goal_chart
            )

        # ----------------------------------------------------
        # Top foods
        # ----------------------------------------------------

        st.divider()

        food_col1, food_col2 = st.columns(2)

        with food_col1:

            st.write(
                "### 🏆 Top Recommended Foods"
            )

            top_foods = (
                history[
                    "Recommended_Food"
                ]
                .value_counts()
                .head(10)
            )

            st.bar_chart(
                top_foods
            )

        # ----------------------------------------------------
        # Category chart
        # ----------------------------------------------------

        with food_col2:

            st.write(
                "### 🍽️ Recommendations by Category"
            )

            category_chart = (
                history[
                    "Food_Category"
                ]
                .value_counts()
                .head(10)
            )

            st.bar_chart(
                category_chart
            )

        # ====================================================
        # RECOMMENDATION TREND
        # ====================================================

        st.divider()

        st.write(
            "### 📅 Recommendation Trend"
        )

        history["Date"] = pd.to_datetime(
            history["Recommendation_Date"],
            errors="coerce"
        ).dt.date

        trend = (
            history
            .groupby("Date")
            .size()
        )

        st.line_chart(
            trend
        )

    else:

        st.info(
            "No recommendation history is available yet. "
            "Generate recommendations from the client page "
            "to populate the dashboard."
        )

    # ========================================================
    # CLIENT DETAILS
    # ========================================================

    st.divider()

    st.subheader(
        "👥 Client Management"
    )

    if history.empty:

        st.info(
            "No clients available."
        )

    else:

        search = st.text_input(
            "🔎 Search Client by Name or Phone"
        )

        client_data = history.copy()

        if search.strip():

            search_text = search.strip().lower()

            client_data = client_data[
                client_data[
                    "Client_Name"
                ]
                .astype(str)
                .str.lower()
                .str.contains(
                    search_text,
                    na=False
                )
                |
                client_data[
                    "Phone_Number"
                ]
                .astype(str)
                .str.contains(
                    search_text,
                    na=False
                )
            ]

        client_columns = [

            "Client_ID",
            "Client_Name",
            "Age",
            "Phone_Number",
            "Gender",
            "Dietary_Type",
            "Health_Goal",
            "Recommended_Food",
            "Recommendation_Date",
            "Recommendation_Score"

        ]

        client_columns = [

            col
            for col in client_columns
            if col in client_data.columns

        ]

        st.dataframe(
            client_data[
                client_columns
            ].sort_values(
                "Recommendation_Date",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # CLIENT SUMMARY
    # ========================================================

    if not history.empty:

        st.divider()

        st.subheader(
            "📋 Client Summary"
        )

        client_summary = (
            history.groupby(
                [
                    "Client_ID",
                    "Client_Name"
                ]
            )
            .agg(
                Recommendations=(
                    "Recommended_Food",
                    "count"
                ),
                Different_Foods=(
                    "Recommended_Food",
                    "nunique"
                ),
                Average_Score=(
                    "Recommendation_Score",
                    "mean"
                )
            )
            .reset_index()
        )

        client_summary[
            "Average_Score"
        ] = client_summary[
            "Average_Score"
        ].round(2)

        st.dataframe(
            client_summary.sort_values(
                "Recommendations",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # FOOD DATA MANAGEMENT
    # ========================================================

    st.divider()

    st.subheader(
        "🍱 Food Dataset Management"
    )

    st.write(
        "View and edit the food recommendation dataset."
    )

    edited_df = st.data_editor(
        df,
        use_container_width=True,
        num_rows="dynamic",
        height=500
    )

    save_food_data = st.button(
        "💾 Save Food Dataset",
        type="primary"
    )

    if save_food_data:

        try:

            edited_df.to_excel(
                DATA_FILE,
                index=False
            )

            st.cache_data.clear()

            st.success(
                "Food dataset successfully updated."
            )

            st.rerun()

        except Exception as e:

            st.error(
                f"Could not save food dataset: {e}"
            )

    # ========================================================
    # FOOD DATA ANALYSIS
    # ========================================================

    st.divider()

    st.subheader(
        "📊 Food Dataset Analysis"
    )

    analysis_col1, analysis_col2 = st.columns(2)

    with analysis_col1:

        st.write(
            "### Food Categories"
        )

        category_data = (
            df["Category"]
            .value_counts()
        )

        st.bar_chart(
            category_data
        )

    with analysis_col2:

        st.write(
            "### Dietary Distribution"
        )

        diet_data = (
            df["Dietary_Type"]
            .value_counts()
        )

        st.bar_chart(
            diet_data
        )

    # ========================================================
    # NUTRITION ANALYSIS
    # ========================================================

    st.divider()

    st.subheader(
        "🥗 Nutritional Analysis"
    )

    nutrition_columns = [

        "Calories",
        "Protein_g",
        "Carbohydrates_g",
        "Sugar_g",
        "Total_Fat_g",
        "Saturated_Fat_g",
        "Fiber_g",
        "Sodium_mg"

    ]

    available_nutrition = [

        col
        for col in nutrition_columns
        if col in df.columns

    ]

    nutrition_summary = (
        df[
            available_nutrition
        ]
        .describe()
        .T
    )

    nutrition_summary = nutrition_summary[
        [
            "mean",
            "min",
            "max"
        ]
    ]

    nutrition_summary = (
        nutrition_summary
        .round(2)
    )

    st.dataframe(
        nutrition_summary,
        use_container_width=True
    )

    # ========================================================
    # AI DATASET ASSISTANT
    # ========================================================

    st.divider()

    st.subheader(
        "🤖 AI Food Dataset Assistant"
    )

    st.write(
        "Ask questions about the current food dataset."
    )

    ai_question = st.text_area(
        "Enter your question",
        placeholder=(
            "Example: Which vegetarian foods have high protein "
            "and low sugar?"
        )
    )

    ai_button = st.button(
        "🤖 Analyze Dataset"
    )

    if ai_button:

        if ai_question.strip() == "":

            st.warning(
                "Please enter a question."
            )

        else:

            question = ai_question.lower()

            result = df.copy()

            # ----------------------------------------------
            # Vegetarian
            # ----------------------------------------------

            if (
                "vegetarian"
                in question
            ):

                result = result[
                    result[
                        "Dietary_Type"
                    ] == "Vegetarian"
                ]

            elif (
                "non vegetarian"
                in question
                or "non-vegetarian"
                in question
                or "nonveg"
                in question
            ):

                result = result[
                    result[
                        "Dietary_Type"
                    ] == "Non-Vegetarian"
                ]

            # ----------------------------------------------
            # High protein
            # ----------------------------------------------

            if (
                "high protein"
                in question
            ):

                protein_limit = df[
                    "Protein_g"
                ].quantile(0.75)

                result = result[
                    result[
                        "Protein_g"
                    ] >= protein_limit
                ]

            # ----------------------------------------------
            # High fiber
            # ----------------------------------------------

            if (
                "high fiber"
                in question
            ):

                fiber_limit = df[
                    "Fiber_g"
                ].quantile(0.75)

                result = result[
                    result[
                        "Fiber_g"
                    ] >= fiber_limit
                ]

            # ----------------------------------------------
            # Low sugar
            # ----------------------------------------------

            if (
                "low sugar"
                in question
            ):

                sugar_limit = df[
                    "Sugar_g"
                ].quantile(0.25)

                result = result[
                    result[
                        "Sugar_g"
                    ] <= sugar_limit
                ]

            # ----------------------------------------------
            # Low sodium
            # ----------------------------------------------

            if (
                "low sodium"
                in question
            ):

                sodium_limit = df[
                    "Sodium_mg"
                ].quantile(0.25)

                result = result[
                    result[
                        "Sodium_mg"
                    ] <= sodium_limit
                ]

            # ----------------------------------------------
            # Low saturated fat
            # ----------------------------------------------

            if (
                "low saturated fat"
                in question
            ):

                fat_limit = df[
                    "Saturated_Fat_g"
                ].quantile(0.25)

                result = result[
                    result[
                        "Saturated_Fat_g"
                    ] <= fat_limit
                ]

            # ----------------------------------------------
            # Highest protein
            # ----------------------------------------------

            if (
                "highest protein"
                in question
            ):

                result = result.sort_values(
                    "Protein_g",
                    ascending=False
                )

            # ----------------------------------------------
            # Lowest sugar
            # ----------------------------------------------

            elif (
                "lowest sugar"
                in question
            ):

                result = result.sort_values(
                    "Sugar_g",
                    ascending=True
                )

            # ----------------------------------------------
            # Results
            # ----------------------------------------------

            st.write(
                "### 🤖 AI Dataset Analysis"
            )

            if result.empty:

                st.warning(
                    "No foods match the requested conditions."
                )

            else:

                st.success(
                    f"Found {len(result)} matching food(s)."
                )

                ai_columns = [

                    "Food_Name",
                    "Category",
                    "Dietary_Type",
                    "Calories",
                    "Protein_g",
                    "Fiber_g",
                    "Sugar_g",
                    "Sodium_mg"

                ]

                ai_columns = [

                    col
                    for col in ai_columns
                    if col in result.columns

                ]

                st.dataframe(
                    result[
                        ai_columns
                    ].head(15),
                    use_container_width=True,
                    hide_index=True
                )

                # AI explanation

                avg_protein = result[
                    "Protein_g"
                ].mean()

                avg_sugar = result[
                    "Sugar_g"
                ].mean()

                avg_fiber = result[
                    "Fiber_g"
                ].mean()

                st.info(
                    f"""
                    **Dataset-based suggestion:**

                    The matching foods have an average protein
                    value of **{avg_protein:.2f} g**, average fiber
                    of **{avg_fiber:.2f} g**, and average sugar of
                    **{avg_sugar:.2f} g** per listed serving.

                    The results above are generated from the
                    current food dataset.
                    """
                )

    # ========================================================
    # DOWNLOAD DATA
    # ========================================================

    st.divider()

    st.subheader(
        "📥 Download Data"
    )

    download_col1, download_col2 = st.columns(2)

    with download_col1:

        food_excel = df.to_excel(
            index=False
        )

        st.download_button(
            label="📥 Download Food Dataset",
            data=food_excel,
            file_name="Food_Dataset.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            )
        )

    with download_col2:

        if not history.empty:

            history_excel = history.to_excel(
                index=False
            )

            st.download_button(
                label="📥 Download Client History",
                data=history_excel,
                file_name="Client_Recommendation_History.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                )
            )


# ============================================================
# CLIENT RECOMMENDATION PAGE
# ============================================================

def recommendation_page():

    st.title(
        "🍱 Personalized Indian Food Recommendation System"
    )

    st.write(
        "Personalized nutrition-based food recommendations "
        "using Machine Learning and nutritional preferences."
    )

    # ========================================================
    # SIDEBAR
    # ========================================================

    st.sidebar.header(
        "👤 Client Information"
    )

    client_name = st.sidebar.text_input(
        "Client Name"
    )

    age = st.sidebar.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=22
    )

    phone_number = st.sidebar.text_input(
        "Phone Number"
    )

    address = st.sidebar.text_area(
        "Address"
    )

    gender = st.sidebar.selectbox(
        "Gender",
        [
            "Prefer not to say",
            "Male",
            "Female",
            "Other"
        ]
    )

    st.sidebar.header(
        "🍽️ Food Preferences"
    )

    dietary_type = st.sidebar.selectbox(
        "Dietary Type",
        [
            "All",
            "Vegetarian",
            "Non-Vegetarian"
        ]
    )

    health_goal = st.sidebar.selectbox(
        "Health Goal",
        [
            "General Healthy Eating",
            "High Protein",
            "High Fiber",
            "Low Sugar",
            "Low Sodium",
            "Low Saturated Fat"
        ]
    )

    st.sidebar.subheader(
        "Nutrition Preferences"
    )

    high_protein = st.sidebar.checkbox(
        "High Protein"
    )

    high_fiber = st.sidebar.checkbox(
        "High Fiber"
    )

    low_sugar = st.sidebar.checkbox(
        "Low Sugar"
    )

    low_sodium = st.sidebar.checkbox(
        "Low Sodium"
    )

    low_saturated_fat = st.sidebar.checkbox(
        "Low Saturated Fat"
    )

    categories = [
        "All"
    ] + sorted(
        df[
            "Category"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    category = st.sidebar.selectbox(
        "Food Category",
        categories
    )

    top_n = st.sidebar.slider(
        "Number of Recommendations",
        min_value=5,
        max_value=20,
        value=10
    )

    # ========================================================
    # APPLY HEALTH GOAL
    # ========================================================

    if health_goal == "High Protein":

        high_protein = True

    elif health_goal == "High Fiber":

        high_fiber = True

    elif health_goal == "Low Sugar":

        low_sugar = True

    elif health_goal == "Low Sodium":

        low_sodium = True

    elif health_goal == "Low Saturated Fat":

        low_saturated_fat = True

    # ========================================================
    # CLIENT HISTORY
    # ========================================================

    history = load_history()

    previous_foods = []

    client_exists = False

    client_history = pd.DataFrame()

    if phone_number.strip() != "":

        if not history.empty:

            client_history = history[
                history[
                    "Phone_Number"
                ]
                .astype(str)
                .str.strip()
                == phone_number.strip()
            ]

            if not client_history.empty:

                client_exists = True

                previous_foods = (
                    client_history[
                        "Recommended_Food"
                    ]
                    .dropna()
                    .unique()
                    .tolist()
                )

    # ========================================================
    # RETURNING CLIENT
    # ========================================================

    recommendation_type = (
        "New Recommendation"
    )

    if client_exists:

        st.warning(
            f"⚠️ Returning client detected. "
            f"{len(previous_foods)} food(s) have previously "
            f"been recommended."
        )

        st.write(
            "**Previously recommended foods:**"
        )

        st.write(
            ", ".join(previous_foods)
        )

        recommendation_type = st.radio(
            "What would you like to do?",
            [
                "Show Different Foods",
                "Show Same Foods Again"
            ]
        )

    # ========================================================
    # GENERATE
    # ========================================================

    st.divider()

    generate = st.button(
        "🔎 Generate Food Recommendations",
        type="primary",
        use_container_width=True
    )

    if generate:

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if client_name.strip() == "":

            st.error(
                "Please enter the client name."
            )

            return

        if phone_number.strip() == "":

            st.error(
                "Please enter the phone number."
            )

            return

        if address.strip() == "":

            st.error(
                "Please enter the address."
            )

            return

        # ----------------------------------------------------
        # EXCLUDED FOODS
        # ----------------------------------------------------

        excluded_foods = []

        if (
            client_exists
            and recommendation_type
            == "Show Different Foods"
        ):

            excluded_foods = previous_foods

        # ----------------------------------------------------
        # RECOMMEND
        # ----------------------------------------------------

        recommendations = get_recommendations(

            df,

            high_protein=high_protein,

            high_fiber=high_fiber,

            low_sugar=low_sugar,

            low_sodium=low_sodium,

            low_saturated_fat=low_saturated_fat,

            dietary_type=dietary_type,

            category=category,

            excluded_foods=excluded_foods,

            top_n=top_n

        )

        if recommendations.empty:

            st.error(
                "No suitable foods were found."
            )

            return

        # ----------------------------------------------------
        # CLIENT ID
        # ----------------------------------------------------

        client_id = generate_client_id(
            phone_number
        )

        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        st.success(
            "Food recommendations generated successfully!"
        )

        # ----------------------------------------------------
        # CLIENT INFORMATION
        # ----------------------------------------------------

        st.subheader(
            "👤 Client Information"
        )

        info1, info2, info3 = st.columns(3)

        with info1:

            st.write(
                f"**Client ID:** {client_id}"
            )

            st.write(
                f"**Name:** {client_name}"
            )

        with info2:

            st.write(
                f"**Age:** {age}"
            )

            st.write(
                f"**Phone:** {phone_number}"
            )

        with info3:

            st.write(
                f"**Diet:** {dietary_type}"
            )

            st.write(
                f"**Health Goal:** {health_goal}"
            )

        # ----------------------------------------------------
        # TOP FOOD
        # ----------------------------------------------------

        top_food = recommendations.iloc[0]

        st.subheader(
            "⭐ Top Recommendation"
        )

        top1, top2, top3, top4 = st.columns(4)

        with top1:

            st.metric(
                "Food",
                top_food["Food_Name"]
            )

        with top2:

            st.metric(
                "Protein",
                f'{top_food["Protein_g"]:.1f} g'
            )

        with top3:

            st.metric(
                "Fiber",
                f'{top_food["Fiber_g"]:.1f} g'
            )

        with top4:

            st.metric(
                "Score",
                f'{top_food["Personalized_Score"]:.1f}'
            )

        # ----------------------------------------------------
        # RECOMMENDATION TABLE
        # ----------------------------------------------------

        st.subheader(
            "🍽️ Recommended Foods"
        )

        display_columns = [

            "Food_Name",
            "Dietary_Type",
            "Category",
            "Serving_Size",
            "Calories",
            "Protein_g",
            "Fiber_g",
            "Sugar_g",
            "Sodium_mg",
            "Personalized_Score",
            "Food_Profile"

        ]

        available_columns = [

            col
            for col in display_columns
            if col in recommendations.columns

        ]

        display_df = recommendations[
            available_columns
        ].copy()

        display_df = display_df.rename(

            columns={

                "Food_Name": "Food",
                "Dietary_Type": "Diet",
                "Serving_Size": "Serving",
                "Protein_g": "Protein (g)",
                "Fiber_g": "Fiber (g)",
                "Sugar_g": "Sugar (g)",
                "Sodium_mg": "Sodium (mg)",
                "Personalized_Score": "Score",
                "Food_Profile": "Profile"

            }

        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # DETAILS
        # ----------------------------------------------------

        st.subheader(
            "📋 Food Details"
        )

        for _, row in recommendations.iterrows():

            with st.expander(
                f'{row["Food_Name"]} — '
                f'Score: '
                f'{row["Personalized_Score"]:.1f}'
            ):

                d1, d2 = st.columns(2)

                with d1:

                    st.write(
                        f'**Category:** {row["Category"]}'
                    )

                    st.write(
                        f'**Diet:** {row["Dietary_Type"]}'
                    )

                    st.write(
                        f'**Serving:** {row["Serving_Size"]}'
                    )

                    st.write(
                        f'**Calories:** {row["Calories"]}'
                    )

                    st.write(
                        f'**Protein:** '
                        f'{row["Protein_g"]} g'
                    )

                with d2:

                    st.write(
                        f'**Fiber:** '
                        f'{row["Fiber_g"]} g'
                    )

                    st.write(
                        f'**Sugar:** '
                        f'{row["Sugar_g"]} g'
                    )

                    st.write(
                        f'**Sodium:** '
                        f'{row["Sodium_mg"]} mg'
                    )

                    st.write(
                        f'**Food Profile:** '
                        f'{row["Food_Profile"]}'
                    )

                    st.write(
                        f'**Recommendation Score:** '
                        f'{row["Personalized_Score"]:.2f}'
                    )

        # ----------------------------------------------------
        # SAVE HISTORY
        # ----------------------------------------------------

        recommendation_records = []

        recommendation_date = (
            datetime.now()
            .strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

        for _, row in recommendations.iterrows():

            record = {

                "Client_ID":
                    client_id,

                "Client_Name":
                    client_name,

                "Age":
                    age,

                "Phone_Number":
                    phone_number,

                "Address":
                    address,

                "Gender":
                    gender,

                "Dietary_Type":
                    dietary_type,

                "Health_Goal":
                    health_goal,

                "High_Protein":
                    high_protein,

                "High_Fiber":
                    high_fiber,

                "Low_Sugar":
                    low_sugar,

                "Low_Sodium":
                    low_sodium,

                "Low_Saturated_Fat":
                    low_saturated_fat,

                "Recommendation_Type":
                    recommendation_type,

                "Recommendation_Date":
                    recommendation_date,

                "Recommended_Food":
                    row["Food_Name"],

                "Food_Category":
                    row["Category"],

                "Food_Profile":
                    row["Food_Profile"],

                "Calories":
                    row["Calories"],

                "Protein_g":
                    row["Protein_g"],

                "Fiber_g":
                    row["Fiber_g"],

                "Sugar_g":
                    row["Sugar_g"],

                "Sodium_mg":
                    row["Sodium_mg"],

                "Recommendation_Score":
                    round(
                        row["Personalized_Score"],
                        2
                    )

            }

            recommendation_records.append(
                record
            )

        new_records_df = pd.DataFrame(
            recommendation_records
        )

        save_history(
            new_records_df
        )

        st.success(
            f"✅ {len(new_records_df)} recommendation(s) "
            "saved to client history."
        )


# ============================================================
# APPLICATION ROUTING
# ============================================================

navigation()


if st.session_state.page == "login":

    if st.session_state.admin_logged_in:

        st.session_state.page = "admin"

        st.rerun()

    else:

        admin_login()


elif (
    st.session_state.page == "admin"
    and st.session_state.admin_logged_in
):

    admin_dashboard()


else:

    recommendation_page()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Personalized Indian Food Recommendation System | "
    "Python | Pandas | Scikit-learn | K-Means | Streamlit"
)