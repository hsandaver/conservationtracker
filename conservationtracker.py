import streamlit as st
import io
import pandas as pd
import re
import json
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

def get_bibliographic_details(record):
    """
    Extracts additional bibliographic details from a MARC record:
      - OCLC and Title (as before)
      - Author: from field 100 (or 700 if 100 not available)
      - Publisher: from field 260 $b or, if absent, 264 $b
      - Printer: from field 264 $e if available (if not, returns 'N/A')
      - Date: from field 260 $c or 264 $c
    Returns a dictionary with these values.
    """
    oclc, title = get_record_identifiers(record)
    
    # Author: try field 100, then 700.
    author = "N/A"
    if record.get_fields("100"):
        field100 = record.get_fields("100")[0]
        if 'a' in field100:
            author = field100['a'].strip()
    elif record.get_fields("700"):
        field700 = record.get_fields("700")[0]
        if 'a' in field700:
            author = field700['a'].strip()
    
    # Publisher: try field 260 $b, else try 264 $b.
    publisher = "N/A"
    if record.get_fields("260"):
        field260 = record.get_fields("260")[0]
        if 'b' in field260:
            publisher = field260['b'].strip()
    elif record.get_fields("264"):
        field264 = record.get_fields("264")[0]
        if 'b' in field264:
            publisher = field264['b'].strip()
    
    # Printer: try field 264 $e (if available)
    printer = "N/A"
    if record.get_fields("264"):
        field264 = record.get_fields("264")[0]
        if 'e' in field264:
            printer = field264['e'].strip()
    
    # Date: try field 260 $c, else 264 $c.
    date = "N/A"
    if record.get_fields("260"):
        field260 = record.get_fields("260")[0]
        if 'c' in field260:
            date = field260['c'].strip()
    elif record.get_fields("264"):
        field264 = record.get_fields("264")[0]
        if 'c' in field264:
            date = field264['c'].strip()
    
    return {"OCLC": oclc, "Title": title, "Author": author, "Publisher": publisher, "Printer": printer, "Date": date}

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
    Processes a MARC 583 field and extracts relevant subfields.
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

def process_340_field(field):
    """
    Processes a MARC 340 field and extracts its subfields.
    All subfields are collected; multiple occurrences are concatenated.
    """
    subfield_keys = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', '0', '1', '2', '3', '6', '8']
    result = {}
    for key in subfield_keys:
        subs = field.get_subfields(key)
        if subs:
            result[key] = "; ".join(subs)
        else:
            result[key] = None
    return result

def read_marc_file(file_bytes):
    """
    Reads MARC records from the provided bytes and returns a MARCReader object.
    """
    stream = io.BytesIO(file_bytes)
    return MARCReader(stream)

def extract_conservation_data(reader, field_code: str, show_raw: bool):
    """
    Processes MARC records to extract conservation (583) data.
    For each record, bibliographic details (including author, publisher, and printer) are added.
    Returns a tuple with:
      - rows: list of dictionaries for DataFrame creation
      - total_records: number of MARC records processed
      - extracted_records: number of records with extracted conservation data
      - raw_records: list of raw MARC record strings if show_raw is True
    """
    rows = []
    total_records = 0
    extracted_records = 0
    raw_records = []
    for record in reader:
        total_records += 1
        bib = get_bibliographic_details(record)
        if field_code in record:
            extracted_records += 1
            for field in record.get_fields(field_code):
                data = process_583_field(field)
                data.update(bib)
                rows.append(data)
        if show_raw:
            raw_records.append(record.as_marc())
    return rows, total_records, extracted_records, raw_records

def convert_marc_to_json(file_bytes, field_code: str, show_raw: bool):
    """
    Converts the MARC file into a JSON-like dict for conservation (583) data.
    """
    reader = read_marc_file(file_bytes)
    rows, total, extracted, raw_records = extract_conservation_data(reader, field_code, show_raw)
    json_data = {
        "records": rows,
        "total_records": total,
        "extracted_records": extracted,
        "raw_records": raw_records if show_raw else []
    }
    return json_data

def extract_340_data(reader, show_raw: bool):
    """
    Processes MARC records to extract physical medium (340) data.
    Bibliographic details are also added.
    Returns a tuple with:
      - rows: list of dictionaries for DataFrame creation
      - total_records: number of MARC records processed
      - raw_records: list of raw MARC record strings if show_raw is True
    """
    rows = []
    total_records = 0
    raw_records = []
    for record in reader:
        total_records += 1
        bib = get_bibliographic_details(record)
        if '340' in record:
            for field in record.get_fields('340'):
                data = process_340_field(field)
                data.update(bib)
                rows.append(data)
        if show_raw:
            raw_records.append(record.as_marc())
    return rows, total_records, raw_records

def convert_marc340_to_json(file_bytes, show_raw: bool):
    """
    Converts the MARC file into a JSON-like dict for physical medium (340) data.
    """
    reader = read_marc_file(file_bytes)
    rows, total, raw_records = extract_340_data(reader, show_raw)
    json_data = {
        "records": rows,
        "total_records": total,
        "raw_records": raw_records if show_raw else []
    }
    return json_data

def display_bibliographic_details(json_data):
    """
    Displays unique bibliographic details extracted from the conservation (583) data.
    Now includes Author, Publisher, and Printer details.
    """
    if not json_data["records"]:
        st.info("No bibliographic details available.")
        return

    bib_details = {}
    for record in json_data["records"]:
        key = (record.get("OCLC", "N/A"), record.get("Title", "N/A"), 
               record.get("Author", "N/A"), record.get("Publisher", "N/A"),
               record.get("Printer", "N/A"), record.get("Date", "N/A"))
        if key not in bib_details:
            bib_details[key] = {
                "OCLC": record.get("OCLC", "N/A"),
                "Title": record.get("Title", "N/A"),
                "Author": record.get("Author", "N/A"),
                "Publisher": record.get("Publisher", "N/A"),
                "Printer": record.get("Printer", "N/A"),
                "Date": record.get("Date", "N/A")
            }

    bib_df = pd.DataFrame(list(bib_details.values()))
    st.subheader("Bibliographic Details")
    st.dataframe(bib_df)

def display_physical_medium_data(df_340: pd.DataFrame):
    """
    Groups and displays the 340 field data with human‑readable column names.
    Instead of one row per occurrence, rows are grouped by bibliographic details
    (OCLC and Title) and aggregated. Human‑readable labels are applied.
    """
    mapping = {
        'a': "Material base and configuration",
        'b': "Dimensions",
        'c': "Material applied to surface",
        'd': "Information recording technique",
        'e': "Support",
        'f': "Reduction ratio value",
        'g': "Color content",
        'h': "Location within medium",
        'i': "Technical specifications of medium",
        'j': "Generation",
        'k': "Layout",
        'l': "Binding",
        'm': "Book format",
        'n': "Font size",
        'o': "Polarity",
        'p': "Illustrative content",
        'q': "Reduction ratio designator",
        '0': "Authority record control number",
        '1': "Real World Object URI",
        '2': "Source",
        '3': "Materials specified",
        '6': "Linkage",
        '8': "Field link and sequence number"
    }
    
    # Group by bibliographic details (OCLC and Title)
    group_cols = ["OCLC", "Title", "Author", "Publisher", "Printer", "Date"]
    agg_funcs = {col: (lambda x: "; ".join(x.dropna().unique())) for col in df_340.columns if col not in group_cols}
    df_grouped = df_340.groupby(group_cols, as_index=False).agg(agg_funcs)
    
    # Rename subfield columns to human-readable labels
    for code, label in mapping.items():
        if code in df_grouped.columns:
            df_grouped = df_grouped.rename(columns={code: label})
    
    # Order columns: bibliographic details first, then the rest
    desired_order = group_cols + list(mapping.values())
    existing_order = [col for col in desired_order if col in df_grouped.columns]
    df_grouped = df_grouped[existing_order]
    
    st.subheader("Physical Medium (340) Details")
    st.dataframe(df_grouped)

def create_download_buttons(df: pd.DataFrame):
    """
    Creates CSV and Excel download buttons for the provided DataFrame.
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

def display_raw_records(raw_records, label="Show Raw MARC Records"):
    """
    Displays raw MARC records within an expander for debugging.
    """
    with st.expander(label):
        for idx, raw in enumerate(raw_records, start=1):
            st.text(f"Record {idx}:")
            st.code(raw)

def filter_data(df: pd.DataFrame):
    """
    Provides an interactive slider to filter the DataFrame by Year.
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
    unique_years = df['Year'].nunique() if 'Year' in df.columns else 0
    avg_actions = total_actions / unique_years if unique_years > 0 else total_actions
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Actions", total_actions)
    col2.metric("Unique Years", unique_years)
    col3.metric("Avg Actions/Year", f"{avg_actions:.1f}")

def create_visualizations(df: pd.DataFrame):
    """
    Creates enhanced visualizations for the conservation (583) data.
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
    st.title("📚 MARC Conservation/Action & Physical Medium (340) Tracker")
    st.sidebar.header("Options")
    field_code = st.sidebar.text_input("Conservation Field to Extract", value="583")
    show_raw = st.sidebar.checkbox("Show raw MARC records", value=False)
    show_json = st.sidebar.checkbox("Show JSON output", value=False)
    show_340 = st.sidebar.checkbox("Display 340 Field Data", value=True)
    
    uploaded_file = st.file_uploader("Upload a MARC file (.mrc or .marc)", type=["mrc", "marc"])
    
    if uploaded_file:
        file_bytes = uploaded_file.read()  # Read file bytes once for both conversions
        
        st.write(f"### Converting MARC file to JSON using field `{field_code}`...")
        try:
            # Process conservation (583) data
            json_data_583 = convert_marc_to_json(file_bytes, field_code, show_raw)
            
            if show_json:
                st.subheader("JSON Output for 583 Data")
                st.json(json_data_583)
            
            # Display unique bibliographic details (including author, publisher, printer)
            display_bibliographic_details(json_data_583)
            
            if json_data_583["records"]:
                df_583 = pd.DataFrame(json_data_583["records"])
                df_583['Year'] = df_583['Date'].apply(extract_year)
                df_583 = filter_data(df_583)
                st.dataframe(df_583)
                create_insight_cards(df_583)
                create_visualizations(df_583)
                create_trend_line(df_583)
                create_action_distribution(df_583)
                create_download_buttons(df_583)
                st.success(f"✅ Extraction Complete! Processed {json_data_583['total_records']} record(s), extracted data from {json_data_583['extracted_records']} record(s).")
            else:
                st.warning("⚠️ No conservation data extracted. Please check your MARC file or field input.")
            
            if show_raw and json_data_583["raw_records"]:
                display_raw_records(json_data_583["raw_records"], label="Raw 583 MARC Records")
            
            # Process and display 340 field data if requested
            if show_340:
                st.write("### Extracting and Displaying 340 Field (Physical Medium) Data")
                json_data_340 = convert_marc340_to_json(file_bytes, show_raw)
                if show_json:
                    st.subheader("JSON Output for 340 Data")
                    st.json(json_data_340)
                if json_data_340["records"]:
                    df_340 = pd.DataFrame(json_data_340["records"])
                    display_physical_medium_data(df_340)
                    st.download_button(
                        label="📥 Download 340 Data as CSV",
                        data=df_340.to_csv(index=False).encode('utf-8'),
                        file_name='physical_medium_340.csv',
                        mime='text/csv'
                    )
                else:
                    st.warning("⚠️ No 340 field data extracted.")
                if show_raw and json_data_340["raw_records"]:
                    display_raw_records(json_data_340["raw_records"], label="Raw 340 MARC Records")
                    
        except Exception as e:
            st.error(f"🚨 Oops, something went wrong while processing the file: {e}")
    else:
        st.info("👆 Please upload a MARC file to begin.")

if __name__ == '__main__':
    main()
