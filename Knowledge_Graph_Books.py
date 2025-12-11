import streamlit as st
import os
import sys
import json
from pathlib import Path
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path=Path(__file__).parent / ".env")
BASE_URL = os.getenv("BASE_URL", "")
GPUSTACK_API_KEY = os.getenv("GPUSTACK_API_KEY", "")

st.set_page_config(page_title="Book Knowledge Graph", layout="wide")

# Import utilities (we'll need to create a simplified version)
from utils_KG import (
    add_topics_to_corpus_table,
    build_paper_ego_graph,
    build_full_corpus_graph,
)

# Initialize OpenAI client for LLM
client = OpenAI(
    base_url=BASE_URL,
    api_key=GPUSTACK_API_KEY
)

st.title("📚 Book Knowledge Graph Generator")
st.markdown("Generate a knowledge graph from the Goodreads books dataset.")

# --- 1. Load Goodreads Dataset ---
st.subheader("📊 Load Dataset")

# Path to goodreads CSV
default_csv_path = "/storage/homefs/as23z124/NLP_CAS_M3_project/goodreads_books.csv"

csv_path = st.text_input(
    "Path to goodreads_books.csv",
    value=default_csv_path,
    help="Enter the full path to the goodreads_books.csv file"
)

if csv_path and Path(csv_path).exists():
    try:
        # Load CSV with progress indicator
        with st.spinner("Loading dataset..."):
            df = pd.read_csv(csv_path)
        
        st.success(f"✅ Loaded {len(df)} books from dataset")
        
        # Show preview
        with st.expander("📖 View Dataset Preview"):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Show column info
        with st.expander("ℹ️ Dataset Columns"):
            st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
            st.write(f"**Total rows:** {len(df)}")
        
        # --- 2. Convert to Corpus Format ---
        st.markdown("---")
        st.markdown("### Step 1: Create Corpus")
        st.markdown("Convert the Goodreads dataset into a format suitable for topic extraction.")
        
        # Let user select how many books to process
        max_books = st.number_input(
            "Number of books to process",
            min_value=10,
            max_value=len(df),
            value=min(100, len(df)),
            step=10,
            help="Processing many books may take time. Start with a smaller sample."
        )
        
        # Output directory configuration
        output_name = st.text_input(
            "Output folder name",
            value="goodreads_corpus",
            help="Name for the output folder"
        )
        
        output_location = st.text_input(
            "Output directory location",
            value="/storage/homefs/as23z124/NLP_CAS_M3_project",
            help="Directory where the output folder will be created"
        )
        
        output_path = Path(output_location) / output_name
        st.info(f"Output will be saved to: `{output_path}`")
        
        if st.button("Create Corpus", type="primary"):
            try:
                # Create output directory
                output_path.mkdir(parents=True, exist_ok=True)
                
                # Sample the dataset
                df_sample = df.head(max_books).copy()
                
                # Convert to corpus format (similar to paper corpus)
                corpus_records = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, row in df_sample.iterrows():
                    status_text.text(f"Processing book {idx+1}/{max_books}: {row.get('title', 'Unknown')}")
                    
                    # Parse recommended books
                    recommended_books_raw = str(row.get('recommended_books', ''))
                    recommended_books = []
                    if recommended_books_raw and recommended_books_raw != 'nan':
                        # Try splitting by comma or semicolon
                        recommended_books = [b.strip() for b in recommended_books_raw.replace(';', ',').split(',') if b.strip()]
                    
                    # Parse books in series
                    books_in_series_raw = str(row.get('books_in_series', ''))
                    books_in_series = []
                    if books_in_series_raw and books_in_series_raw != 'nan':
                        books_in_series = [b.strip() for b in books_in_series_raw.replace(';', ',').split(',') if b.strip()]
                    
                    # Create book record
                    record = {
                        "book_id": str(row.get('id', '')),
                        "title": str(row.get('title', 'Untitled')),
                        "author": str(row.get('author', 'Unknown')),
                        "description": str(row.get('description', '')),
                        "link": str(row.get('link', '')),
                        "cover_link": str(row.get('cover_link', '')),
                        "author_link": str(row.get('author_link', '')),
                        "publisher": str(row.get('publisher', '')),
                        "date_published": str(row.get('date_published', '')),
                        "original_title": str(row.get('original_title', '')),
                        "genre_and_votes": str(row.get('genre_and_votes', '')),
                        "series": str(row.get('series', '')),
                        "number_of_pages": row.get('number_of_pages', ''),
                        "isbn": str(row.get('isbn', '')),
                        "isbn13": str(row.get('isbn13', '')),
                        "average_rating": row.get('average_rating', 0),
                        "rating_count": row.get('rating_count', 0),
                        "review_count": row.get('review_count', 0),
                        "settings": str(row.get('settings', '')),
                        "characters": str(row.get('characters', '')),
                        "awards": str(row.get('awards', '')),
                        "recommended_books": recommended_books,
                        "books_in_series": books_in_series,
                    }
                    
                    corpus_records.append(record)
                    progress_bar.progress((idx + 1) / max_books)
                
                # Save as JSONL (corpus format)
                corpus_jsonl = output_path / "corpus_table.jsonl"
                with open(corpus_jsonl, 'w', encoding='utf-8') as f:
                    for record in corpus_records:
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')
                
                # Save as CSV for convenience
                corpus_df = pd.DataFrame(corpus_records)
                corpus_csv = output_path / "corpus_table.csv"
                corpus_df.to_csv(corpus_csv, index=False)
                
                progress_bar.progress(1.0)
                status_text.empty()
                
                st.success(f"""
                ✅ **Corpus Created!**
                - Total books processed: {len(corpus_records)}
                - Output saved to: `{output_path}`
                - Files: corpus_table.jsonl, corpus_table.csv
                """)
                
                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    with open(corpus_jsonl, 'r', encoding='utf-8') as f:
                        st.download_button(
                            label="⬇️ Download JSONL",
                            data=f.read(),
                            file_name="corpus_table.jsonl",
                            mime="application/jsonlines"
                        )
                with col2:
                    with open(corpus_csv, 'rb') as f:
                        st.download_button(
                            label="⬇️ Download CSV",
                            data=f,
                            file_name="corpus_table.csv",
                            mime="text/csv"
                        )
                
            except Exception as e:
                st.error(f"❌ Error creating corpus: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        
        # --- 3. Extract Topics ---
        st.markdown("---")
        st.markdown("### Step 2: Extract Topics with LLM")
        st.markdown("Use AI to extract themes and topics from book descriptions.")
        
        corpus_location = st.text_input(
            "Corpus directory location",
            value=output_location,
            key="corpus_loc",
            help="Directory containing your corpus folder"
        )
        
        if corpus_location:
            corpus_path_parent = Path(corpus_location)
            
            if corpus_path_parent.exists() and corpus_path_parent.is_dir():
                # Find corpus folders
                corpus_folders = sorted(corpus_path_parent.glob("*_corpus"))
                valid_corpora = [f for f in corpus_folders if (f / "corpus_table.jsonl").exists()]
                
                if valid_corpora:
                    st.success(f"✅ Found {len(valid_corpora)} corpus/corpora")
                    
                    corpus_options = {f.name: f for f in valid_corpora}
                    selected_corpus_name = st.selectbox(
                        "Select corpus for topic extraction",
                        options=list(corpus_options.keys()),
                        key="topic_corpus_select"
                    )
                    
                    selected_corpus = corpus_options[selected_corpus_name]
                    st.info(f"📍 Selected: `{selected_corpus}`")
                    
                    if st.button("🤖 Extract Topics", type="secondary"):
                        topic_progress = st.progress(0)
                        topic_status = st.empty()
                        
                        try:
                            def update_progress(current, total):
                                topic_progress.progress(current / total)
                            
                            def update_status(message):
                                topic_status.text(message)
                            
                            add_topics_to_corpus_table(
                                output_root=selected_corpus,
                                client=client,
                                inplace=False,
                                model="gpt-oss-120b",
                                progress_callback=update_progress,
                                status_callback=update_status
                            )
                            
                            topic_progress.progress(1.0)
                            topic_status.empty()
                            
                            st.success("""
                            ✅ **Topic Extraction Complete!**
                            - Topics have been extracted and added to the corpus
                            - Output file: `corpus_table.with_topics.jsonl`
                            """)
                            
                            topics_file = selected_corpus / "corpus_table.with_topics.jsonl"
                            if topics_file.exists():
                                with open(topics_file, "r", encoding="utf-8") as f:
                                    topics_data = f.read()
                                
                                st.download_button(
                                    label="⬇️ Download Corpus with Topics (JSONL)",
                                    data=topics_data,
                                    file_name="corpus_table.with_topics.jsonl",
                                    mime="application/jsonlines"
                                )
                        
                        except Exception as e:
                            st.error(f"❌ Topic extraction failed: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                else:
                    st.warning("⚠️ No valid corpus found. Please complete Step 1 first.")
        
        # --- 4. Build Knowledge Graph ---
        st.markdown("---")
        st.markdown("### Step 3: Build Knowledge Graph")
        st.markdown("Generate network visualization from books with extracted topics.")
        
        kg_location = st.text_input(
            "Knowledge Graph corpus location",
            value=output_location,
            key="kg_location",
            help="Directory containing corpus with topics"
        )
        
        if kg_location:
            kg_path = Path(kg_location)
            
            if kg_path.exists() and kg_path.is_dir():
                kg_corpus_folders = sorted(kg_path.glob("*_corpus"))
                valid_kg_corpora = [f for f in kg_corpus_folders if (f / "corpus_table.with_topics.jsonl").exists()]
                
                if valid_kg_corpora:
                    st.success(f"✅ Found {len(valid_kg_corpora)} corpus/corpora with topics")
                    
                    kg_corpus_options = {f.name: f for f in valid_kg_corpora}
                    selected_kg_corpus_name = st.selectbox(
                        "Select corpus for visualization",
                        options=list(kg_corpus_options.keys()),
                        key="kg_corpus_select"
                    )
                    
                    selected_kg_corpus = kg_corpus_options[selected_kg_corpus_name]
                    st.info(f"📍 Selected: `{selected_kg_corpus}`")
                    
                    # Load corpus data
                    topics_file = selected_kg_corpus / "corpus_table.with_topics.jsonl"
                    
                    all_records = []
                    with open(topics_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            all_records.append(json.loads(line))
                    
                    # Book selection for ego graph
                    book_display_options = {}
                    for rec in all_records:
                        book_id = rec.get("book_id", "unknown")
                        title = rec.get("title", "Untitled")
                        display_label = f"{book_id} - {title}"
                        book_display_options[display_label] = rec
                    
                    selected_display_label = st.selectbox(
                        "Select book to visualize ego graph",
                        options=list(book_display_options.keys()),
                        key="book_select"
                    )
                    
                    selected_book = book_display_options[selected_display_label]
                    selected_book_id = selected_book.get("book_id", "unknown")
                    book_title = selected_book.get("title", "Untitled")
                    st.write(f"**Selected:** {book_title}")
                    
                    # Options for ego graph
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        include_authors = st.checkbox("Authors", value=True, key="ego_authors")
                    with col2:
                        include_topics = st.checkbox("Topics", value=True, key="ego_topics")
                    with col3:
                        include_cited_papers = st.checkbox("Related Books", value=False, key="ego_cited_papers")
                    
                    if st.button("Generate Ego Graph", type="primary"):
                        try:
                            with st.spinner("Building ego graph..."):
                                nx_graph, pyvis_net = build_paper_ego_graph(
                                    selected_book,
                                    all_records=all_records,
                                    include_topics=include_topics,
                                    include_authors=include_authors,
                                    include_cited_papers=include_cited_papers
                                )
                                
                                html_file = selected_kg_corpus / f"{selected_book_id}_ego_graph.html"
                                pyvis_net.save_graph(str(html_file))
                                
                                with open(html_file, 'r', encoding='utf-8') as f:
                                    html_content = f.read()
                                
                                st.success("✅ Ego graph generated!")
                                
                                import streamlit.components.v1 as components
                                components.html(html_content, height=650, scrolling=True)
                                
                                st.markdown("**Graph Statistics:**")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Nodes", nx_graph.number_of_nodes())
                                with col2:
                                    st.metric("Edges", nx_graph.number_of_edges())
                                with col3:
                                    st.metric("Authors", len([n for n in nx_graph.nodes if str(n).startswith("author:")]))
                                
                                st.download_button(
                                    label="💾 Download HTML",
                                    data=html_content,
                                    file_name=f"{selected_book_id}_ego_graph.html",
                                    mime="text/html"
                                )
                        
                        except Exception as e:
                            st.error(f"❌ Error generating ego graph: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                    
                    # Full Corpus Graph
                    st.markdown("---")
                    st.markdown("#### Full Corpus Graph")
                    st.markdown("Visualize all books, authors, and topics together.")
                    
                    # Book selection expander
                    with st.expander("🔍 Advanced: Select Specific Books (Optional)", expanded=False):
                        st.markdown("By default, all books are included.")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            select_all = st.button("✅ Select All", key="select_all_books")
                        with col_b:
                            deselect_all = st.button("❌ Deselect All", key="deselect_all_books")
                        
                        if "selected_books" not in st.session_state:
                            st.session_state.selected_books = {rec.get("book_id"): True for rec in all_records}
                        
                        if select_all:
                            st.session_state.selected_books = {rec.get("book_id"): True for rec in all_records}
                        
                        if deselect_all:
                            st.session_state.selected_books = {rec.get("book_id"): False for rec in all_records}
                        
                        num_cols = 3
                        cols = st.columns(num_cols)
                        
                        for idx, rec in enumerate(all_records):
                            book_id = rec.get("book_id", f"unknown_{idx}")
                            title = rec.get("title", "Untitled")
                            
                            with cols[idx % num_cols]:
                                display_text = f"{book_id} - {title[:40]}..." if len(title) > 40 else f"{book_id} - {title}"
                                
                                is_selected = st.checkbox(
                                    display_text,
                                    value=st.session_state.selected_books.get(book_id, True),
                                    key=f"book_checkbox_{book_id}"
                                )
                                st.session_state.selected_books[book_id] = is_selected
                        
                        selected_count = sum(st.session_state.selected_books.values())
                        st.info(f"Selected: {selected_count} / {len(all_records)} books")
                    
                    # Full graph options
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        include_authors_full = st.checkbox("Authors", value=True, key="full_authors")
                    with col2:
                        include_topics_full = st.checkbox("Topics", value=True, key="full_topics")
                    with col3:
                        include_cited_papers_full = st.checkbox("Related Books", value=False, key="full_cited_papers")
                    with col4:
                        min_confidence = st.slider("Min Topic Confidence", 0.0, 1.0, 0.0, 0.1, key="min_conf")
                    
                    if st.button("🌐 Generate Full Corpus Graph", type="secondary"):
                        try:
                            with st.spinner("Building full corpus graph..."):
                                if "selected_books" in st.session_state:
                                    filtered_records = [
                                        rec for rec in all_records 
                                        if st.session_state.selected_books.get(rec.get("book_id"), True)
                                    ]
                                else:
                                    filtered_records = all_records
                                
                                if not filtered_records:
                                    st.warning("⚠️ No books selected. Please select at least one book.")
                                else:
                                    nx_graph_full, pyvis_net_full = build_full_corpus_graph(
                                        filtered_records,
                                        include_topics=include_topics_full,
                                        include_authors=include_authors_full,
                                        include_cited_papers=include_cited_papers_full,
                                        min_topic_confidence=min_confidence
                                    )
                                    
                                    html_file_full = selected_kg_corpus / "full_corpus_graph.html"
                                    pyvis_net_full.save_graph(str(html_file_full))
                                    
                                    with open(html_file_full, 'r', encoding='utf-8') as f:
                                        html_content_full = f.read()
                                    
                                    st.success(f"✅ Full corpus graph generated with {len(filtered_records)} books!")
                                    
                                    import streamlit.components.v1 as components
                                    components.html(html_content_full, height=850, scrolling=True)
                                    
                                    st.markdown("**Graph Statistics:**")
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Total Nodes", nx_graph_full.number_of_nodes())
                                    with col2:
                                        st.metric("Total Edges", nx_graph_full.number_of_edges())
                                    with col3:
                                        st.metric("Books", len([n for n in nx_graph_full.nodes if not str(n).startswith(("author:", "topic:"))]))
                                    with col4:
                                        st.metric("Authors", len([n for n in nx_graph_full.nodes if str(n).startswith("author:")]))
                                    
                                    st.download_button(
                                        label="💾 Download Full Graph HTML",
                                        data=html_content_full,
                                        file_name="full_corpus_graph.html",
                                        mime="text/html"
                                    )
                        
                        except Exception as e:
                            st.error(f"❌ Error generating full corpus graph: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                
                else:
                    st.warning("⚠️ No corpus with topics found. Please complete Step 2 first.")
    
    except Exception as e:
        st.error(f"❌ Error loading CSV: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

elif csv_path:
    st.error("❌ File not found. Please check the path.")

# Information section
st.markdown("---")
st.subheader("ℹ️ About Book Knowledge Graph")
st.markdown("""
This tool will:
1. Load books from the Goodreads dataset
2. Extract themes and topics from book descriptions using AI
3. Generate a knowledge graph showing relationships between books, authors, and topics
4. Allow you to explore and export the results

**Data Source:**
- Goodreads Books CSV dataset
- Book descriptions are used for topic extraction
""")
