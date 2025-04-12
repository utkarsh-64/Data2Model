import streamlit as st
import pandas as pd
import sweetviz as sv
import io

def export_section():
    st.title("Data Export")
    
    if st.session_state.processed_df is None:
        st.warning("No data to export!")
        return
    
    with st.expander("📤 Export Options", expanded=True):
        export_format = st.radio(
            "Select Export Format",
            ["CSV", "Excel"],
            horizontal=True
        )
        
        filename = st.text_input("File name", "processed_data")
        
        if export_format == "CSV":
            csv = st.session_state.processed_df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                f"{filename}.csv",
                "text/csv"
            )
        else:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                st.session_state.processed_df.to_excel(writer, index=False)
            st.download_button(
                "Download Excel",
                output.getvalue(),
                f"{filename}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with st.expander("🔍 Final Data Preview", expanded=True):
        st.dataframe(st.session_state.processed_df, use_container_width=True)
        st.write(f"Processed Data Shape: {st.session_state.processed_df.shape}")

    with st.expander("📊 Analysis Report", expanded=True):
        if st.button("Generate Sweetviz Report"):
            try:
                report = sv.compare(
                    [st.session_state.raw_df, "Raw Data"],
                    [st.session_state.processed_df, "Processed Data"]
                )
                report.show_html("report.html")
                with open("report.html", "r") as f:
                    html_report = f.read()
                st.components.v1.html(html_report, width=1000, height=800)
            except Exception as e:
                st.error(f"Report generation failed: {str(e)}")    