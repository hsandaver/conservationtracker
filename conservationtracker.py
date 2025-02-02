import streamlit as st
import io
import pandas as pd
import re
from pymarc import MARCReader
import plotly.express as px

def apply_custom_css():
    st.markdown("""
        <style>
        body {
            background-color: #f0f2f6;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 4px;
        }
        .reportview-container .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)

def get_record_identifiers(record):
    """
    Extracts the OCLC number and Title from the MARC record.
    - OCLC number is usually found in field '035' subfield 'a' containing "OCoLC".
    - Title is taken from field '245' using subfields 'a' and optionally 'b'.
    """
    oclc = "N/A"
    for field in record.get_fields('035'):
        if 'a' in field:
            value = field['a']
            if "OCoLC" in value:
                oclc = value.strip()
                break
    title = "N/A"
    fields_245 = record.get_fields('245')
    if fields_245:
        field_245 = fields_245[0]
        title_parts = []
        if 'a' in field_245:
            title_parts.append(field_245['a'].strip())
        if 'b' in field_245:
            title_parts.append(field_245['b'].strip())
        if title_parts:
            title = " ".join(title_parts)
    return oclc, title

def extract_year(date_str: str):
    """
    Extracts the first 4-digit year from the date string.
    Assumes dates are in ISO 8601 (e.g., YYYYMMDD) without hyphens.
    """
    if date_str:
        match = re.search(r'(\d{4})', date_str)
        if match:
            return int(match.group(1))
    return None

def process_583_field(field):
    """
    Processes a MARC 583 field and extracts relevant subfields as per the LOC spec.
    Mandatory subfields: $a (Action), $c (Date), $2 (Source of Term), $5 (Institution).
    Recommended subfield: $l (Status).
    Also extracts:
      - $3: Materials Specified
      - $b: Action Identification
      - $f: Authorization
      - $h: Jurisdiction
      - $i: Method of Action
      - $j: Site of Action
      - $k: Action Agent
      - $u: Uniform Resource Identifier
      - $x: Nonpublic Note
      - $z: Public Note
    For subfields that might appear multiple times (like $l for Status), concatenates all occurrences using a semicolon.
    """
    result = {
        'Materials': field.get('3', 'N/A'),
        'Action': field.get('a', 'N/A'),
        'Action Identification': field.get('b', ''),
        'Date': field.get('c', 'N/A'),
        'Status': "; ".join(field.get_subfields('l')) if field.get_subfields('l') else "N/A",
        'Authorization': field.get('f', ''),
        'Jurisdiction': field.get('h', ''),
        'Method': field.get('i', ''),
        'Site': field.get('j', ''),
        'Agent': field.get('k', ''),
        'URI': field.get('u', ''),
        'Nonpublic Note': field.get('x', ''),
        'Public Note': field.get('z', ''),
        'Source': field.get('2', 'N/A'),
        'Institution': field.get('5', 'N/A'),
    }
    for key in result:
        if isinstance(result[key], str):
            result[key] = result[key].strip()
    return result

def read_marc_file(uploaded_file) -> MARCReader:
    """
    Reads the uploaded MARC file and returns a MARCReader object.
    """
    file_bytes = uploaded_file.read()
    stream = io.BytesIO(file_bytes)
    return MARCReader(stream)

def extract_conservation_data(reader, field_code: str, show_raw: bool):
    """
    Processes MARC records to extract detailed conservation data from the specified field.
    Returns:
        rows (list): A list of dictionaries (one per field instance) ready for DataFrame creation.
        total_records (int): Total number of MARC records processed.
        extracted_records (int): Number of records from which data was extracted.
        raw_records (list): List of raw MARC record strings (if show_raw is True).
    """
    rows = []
    total_records = 0
    extracted_records = 0
    raw_records = []
    for record in reader:
        total_records += 1
        oclc, title = get_record_identifiers(record)
        if field_code in record:
            extracted_records += 1
            for field in record.get_fields(field_code):
                data = process_583_field(field)
                data['OCLC'] = oclc
                data['Title'] = title
                rows.append(data)
        if show_raw:
            raw_records.append(record.as_marc())
    return rows, total_records, extracted_records, raw_records

def create_download_buttons(df: pd.DataFrame):
    """
    Creates CSV and Excel download buttons for the DataFrame.
    """
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download as CSV",
        data=csv_data,
        file_name='conservation_actions.csv',
        mime='text/csv'
    )
    excel_io = io.BytesIO()
    with pd.ExcelWriter(excel_io, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='ConservationActions')
    excel_io.seek(0)
    st.download_button(
        label="📥 Download as Excel",
        data=excel_io,
        file_name='conservation_actions.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

def display_raw_records(raw_records):
    """
    Displays raw MARC records within an expander for debugging.
    """
    with st.expander("Show Raw MARC Records"):
        for idx, raw in enumerate(raw_records, start=1):
            st.text(f"Record {idx}:")
            st.code(raw)

def filter_data(df: pd.DataFrame):
    """
    Provides an interactive slider to filter the DataFrame by Year.
    If only one year is present, a message is shown instead.
    """
    if 'Date' in df.columns:
        df['Year'] = df['Date'].apply(extract_year)
    if 'Year' in df.columns and not df['Year'].isnull().all():
        min_year = int(df['Year'].min())
        max_year = int(df['Year'].max())
        if min_year == max_year:
            st.sidebar.write(f"Only one year found: {min_year}. No filtering applied.")
            return df
        else:
            selected_range = st.sidebar.slider("Filter by Year", min_year, max_year, (min_year, max_year))
            df = df[(df['Year'] >= selected_range[0]) & (df['Year'] <= selected_range[1])]
    return df

def create_insight_cards(df: pd.DataFrame):
    """
    Creates insight cards that display aggregated metrics.
    """
    st.markdown("### Data Insights")
    total_actions = len(df)
    if 'Year' in df.columns:
        unique_years = df['Year'].nunique()
    else:
        unique_years = 0
    avg_actions = total_actions / unique_years if unique_years > 0 else total_actions
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Actions", total_actions)
    col2.metric("Unique Years", unique_years)
    col3.metric("Avg Actions/Year", f"{avg_actions:.1f}")

def create_visualizations(df: pd.DataFrame):
    """
    Creates enhanced visualizations:
      - Bar Chart: Conservation Actions by Year.
      - Pie Chart: Distribution of Condition Status.
    """
    st.subheader("Enhanced Visualizations")
    if 'Date' in df.columns:
        df['Year'] = df['Date'].apply(extract_year)
    valid_years = df.dropna(subset=['Year'])
    if not valid_years.empty:
        year_counts = valid_years.groupby('Year').size().reset_index(name='Count')
        fig = px.bar(year_counts, x='Year', y='Count', title='Conservation Actions by Year')
        if year_counts.shape[0] == 1:
            single_year = year_counts.iloc[0]['Year']
            fig.update_layout(xaxis=dict(range=[single_year - 1, single_year + 1]))
        st.plotly_chart(fig)
    else:
        st.info("No valid year data found for visualization.")
    
    if 'Status' in df.columns and not df['Status'].empty:
        status_counts = df['Status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        if not status_counts.empty:
            fig2 = px.pie(status_counts, names='Status', values='Count', title='Distribution of Condition Status')
            st.plotly_chart(fig2)

def create_trend_line(df: pd.DataFrame):
    """
    Creates a line chart showing the trend of conservation actions over time.
    """
    valid_years = df.dropna(subset=['Year'])
    if not valid_years.empty:
        year_counts = valid_years.groupby('Year').size().reset_index(name='Count')
        fig = px.line(year_counts, x='Year', y='Count', markers=True, title="Trend of Conservation Actions Over Years")
        st.plotly_chart(fig)

def create_action_distribution(df: pd.DataFrame):
    """
    Creates a bar chart for the distribution of Action types.
    """
    if 'Action' in df.columns and not df['Action'].empty:
        action_counts = df['Action'].value_counts().reset_index()
        action_counts.columns = ['Action', 'Count']
        fig = px.bar(action_counts, x='Action', y='Count', title="Distribution of Action Types")
        st.plotly_chart(fig)

def main():
    apply_custom_css()
    st.title("📚 MARC Conservation/Action Tracker")
    st.sidebar.header("Options")
    field_code = st.sidebar.text_input("MARC Field to Extract", value="583")
    show_raw = st.sidebar.checkbox("Show raw MARC records", value=False)
    uploaded_file = st.file_uploader("Upload a MARC file (.mrc or .marc)", type=["mrc", "marc"])

    if uploaded_file:
        st.write(f"### Extracting data from MARC `{field_code}` Fields...")
        try:
            reader = read_marc_file(uploaded_file)
            rows, total, extracted, raw_records = extract_conservation_data(reader, field_code, show_raw)
            if rows:
                df = pd.DataFrame(rows)
                df['Year'] = df['Date'].apply(extract_year)  # Ensure Year column exists.
                df = filter_data(df)  # Apply interactive filtering.
                st.dataframe(df)
                create_insight_cards(df)
                create_visualizations(df)
                create_trend_line(df)
                create_action_distribution(df)
                create_download_buttons(df)
                st.success(f"✅ Extraction Complete! Processed {total} record(s), extracted data from {extracted} record(s).")
            else:
                st.warning("⚠️ No data extracted from the specified field. Please check your MARC file or field input, nyah~!")
            if show_raw and raw_records:
                display_raw_records(raw_records)
        except Exception as e:
            st.error(f"🚨 Oops, something went wrong while processing the file: {e}")
    else:
        st.info("👆 Please upload a MARC file to begin.")

if __name__ == '__main__':
    main()