import subprocess
import socket
import time
import sys
import os
import requests
import xml.etree.ElementTree as ET
import re
import json
from pathlib import Path
import pandas as pd
from openai import OpenAI

GROBID_HOST = "127.0.0.1"
GROBID_PORT = 8070  # Default Grobid port
GROBID_URL = f"http://{GROBID_HOST}:{GROBID_PORT}/api/processFulltextDocument"
GROBID_CONTAINER = os.getenv("GROBID_CONTAINER", "/storage/research/dsl_shared/solutions/ondemand/text_lab/container/grobid_0.8.2.sif")
GROBID_TMP = os.getenv("GROBID_TMP", os.path.join(os.getcwd(), "grobid-tmp"))

# TEI namespace
NS = {"tei": "http://www.tei-c.org/ns/1.0"}

# System prompt for LLM topic extraction
SYSTEM_PROMPT = """
You are an expert book curator and literary analyst.

Your task: read the provided book information (title and description) 
and extract high-quality, hierarchical topics that capture the book's themes, genres, and content.

Output requirements:
- Return ONLY valid JSON (no explanation, no intro, no markdown).
- Use this exact schema:

{
  "topics": [
    { 
      "category": "string",
      "label": "string", 
      "confidence": 0.0, 
      "rationale": "string" 
    }
  ]
}

Rules:
- Provide 3–8 topics.
- Each topic has TWO levels:
  * "category": A BROAD genre or theme area (e.g., "Fiction", "Biography", "Self-Help", "Science Fiction", "Historical Fiction", "Business")
  * "label": A SPECIFIC topic or theme within that category (2–6 words)
- Categories should be general enough that multiple books could share them.
- Labels must be specific to the content and themes of this book.
- Avoid overly generic words ("story", "narrative", "chapters", "pages", "writing").
- Avoid meta-topics ("author's note", "acknowledgments", "book series").
- Focus on themes, subject matter, settings, and key concepts.
- Confidence should be between 0 and 1.
- Rationales must be one sentence each.

Example output:
{
  "topics": [
    { "category": "Science Fiction", "label": "Dystopian Society", "confidence": 0.95, "rationale": "Book explores a totalitarian future world." },
    { "category": "Philosophy", "label": "Individual Freedom", "confidence": 0.85, "rationale": "Central theme examines personal autonomy under state control." }
  ]
}
"""

class GrobidError(Exception):
    """Custom exception for Grobid-related errors."""
    pass

def _port_open(host, port):
    """Check if a port is open."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

def ensure_grobid_server():
    """Ensure Grobid server is running, start it if not.
    
    Raises:
        GrobidError: If Grobid cannot be started.
    """
    # 0. Fast path ─ already up?
    if _port_open(GROBID_HOST, GROBID_PORT):
        return

    # 1. Make sure we have a writable tmp dir
    os.makedirs(GROBID_TMP, exist_ok=True)
    
    # 2. Check if apptainer/singularity is available
    apptainer_cmd = None
    for cmd in ["apptainer", "singularity"]:
        if subprocess.run(["which", cmd], capture_output=True).returncode == 0:
            apptainer_cmd = cmd
            break
    
    if apptainer_cmd is None:
        raise GrobidError("`apptainer` or `singularity` not found. Cannot start Grobid container.")

    # 3. Check if container file exists
    if not os.path.exists(GROBID_CONTAINER):
        raise GrobidError(f"Grobid container `{GROBID_CONTAINER}` not found.")

    # 4. Build the command
    grobid_command = [
        apptainer_cmd, "exec",
        "-B", f"{GROBID_TMP}:/opt/grobid/grobid-home/tmp",
        "--env", "GROBID_HOME=/opt/grobid/grobid-home",
        GROBID_CONTAINER,
        "bash", "-c", "cd /opt/grobid && ./grobid-service/bin/grobid-service"
    ]

    # 5. Spawn the Grobid service
    # 5. Spawn the Grobid service
    try:
        subprocess.Popen(
            grobid_command,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
    except Exception as e:
        raise GrobidError(f"Failed to start Grobid container: {e}")
    # 6. Wait (max 60 s) until the TCP port answers
    # 6. Wait (max 60 s) until the TCP port answers
    for _ in range(120):  # Grobid can take longer to start
        if _port_open(GROBID_HOST, GROBID_PORT):
            return
        time.sleep(0.5)

    raise GrobidError("Grobid server failed to start within 60 seconds - check container and logs.")


def grobid_process_pdf_to_xml(pdf_path):
    """
    Send a PDF to Grobid and return the TEI XML as a string.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        str: TEI XML content
        
    Raises:
        GrobidError: If processing fails
    """
    ensure_grobid_server()
    
    with open(pdf_path, "rb") as f:
        response = requests.post(
            GROBID_URL,
            files={"input": f},
            timeout=60
        )
    
    if response.status_code == 200:
        return response.text
    else:
        raise GrobidError(f"Grobid returned status {response.status_code}: {response.text}")


def extract_metadata_fields(tei_xml, pdf_path, paper_id):
    """
    Extract metadata fields from TEI XML.
    
    Args:
        tei_xml: TEI XML string from Grobid
        pdf_path: Path to original PDF
        paper_id: Unique identifier for the paper
        
    Returns:
        dict: Metadata including title, authors, abstract, keywords, DOI, citations, etc.
    """
    root = ET.fromstring(tei_xml.encode('utf-8'))
    
    metadata = {
        "paper_id": paper_id,
        "pdf_path": str(pdf_path)
    }
    
    # Extract title
    title_elem = root.find(".//tei:titleStmt/tei:title[@type='main']", NS)
    metadata["title"] = title_elem.text.strip() if title_elem is not None and title_elem.text else ""
    
    # Extract authors - try both paths
    authors = []
    
    # Path 1: titleStmt (for paper metadata in some cases)
    for author in root.findall(".//tei:titleStmt/tei:author", NS):
        persName = author.find("tei:persName", NS)
        if persName is not None:
            forename = persName.find("tei:forename[@type='first']", NS)
            surname = persName.find("tei:surname", NS)
            if forename is not None and surname is not None:
                authors.append(f"{forename.text} {surname.text}")
    
    # Path 2: sourceDesc/biblStruct (more common for paper authors)
    if not authors:
        for author in root.findall(".//tei:sourceDesc/tei:biblStruct/tei:analytic/tei:author", NS):
            persName = author.find("tei:persName", NS)
            if persName is not None:
                forename = persName.find("tei:forename[@type='first']", NS)
                surname = persName.find("tei:surname", NS)
                if forename is not None and surname is not None:
                    authors.append(f"{forename.text} {surname.text}")
    
    metadata["authors"] = authors
    metadata["n_authors"] = len(authors)
    
    # Extract abstract
    abstract_elem = root.find(".//tei:abstract/tei:div/tei:p", NS)
    metadata["abstract"] = abstract_elem.text.strip() if abstract_elem is not None and abstract_elem.text else ""
    
    # Extract keywords
    keywords = []
    for term in root.findall(".//tei:keywords/tei:term", NS):
        if term.text:
            keywords.append(term.text.strip())
    metadata["keywords"] = keywords
    
    # Extract DOI
    doi_elem = root.find(".//tei:idno[@type='DOI']", NS)
    metadata["DOI"] = doi_elem.text.strip() if doi_elem is not None and doi_elem.text else ""
    
    # Extract journal info
    journal_elem = root.find(".//tei:sourceDesc/tei:biblStruct/tei:monogr/tei:title", NS)
    metadata["journal"] = journal_elem.text.strip() if journal_elem is not None and journal_elem.text else ""
    
    # Extract publication date
    date_elem = root.find(".//tei:sourceDesc/tei:biblStruct/tei:monogr/tei:imprint/tei:date[@type='published']", NS)
    metadata["publication_date"] = date_elem.get("when", "") if date_elem is not None else ""
    
    # Extract citations (cited papers)
    citations = []
    cited_authors = []
    cited_dois = []
    
    for bibl in root.findall(".//tei:text/tei:back//tei:listBibl/tei:biblStruct", NS):
        citation = {}
        
        # Citation ID
        citation["citation_id"] = bibl.get("{http://www.w3.org/XML/1998/namespace}id", "")
        
        # Title
        title_elem = bibl.find(".//tei:analytic/tei:title[@type='main']", NS)
        if title_elem is None:
            title_elem = bibl.find(".//tei:monogr/tei:title", NS)
        citation["title"] = title_elem.text.strip() if title_elem is not None and title_elem.text else ""
        
        # Authors
        citation_authors = []
        for author in bibl.findall(".//tei:analytic/tei:author", NS):
            persName = author.find("tei:persName", NS)
            if persName is not None:
                forename = persName.find("tei:forename[@type='first']", NS)
                middle = persName.find("tei:forename[@type='middle']", NS)
                surname = persName.find("tei:surname", NS)
                
                if forename is not None and surname is not None:
                    if middle is not None and middle.text:
                        citation_authors.append(f"{forename.text} {middle.text} {surname.text}")
                    else:
                        citation_authors.append(f"{forename.text} {surname.text}")
                elif surname is not None:
                    citation_authors.append(surname.text)
        
        citation["authors"] = citation_authors
        cited_authors.extend(citation_authors)
        
        # DOI
        doi_elem = bibl.find(".//tei:idno[@type='DOI']", NS)
        doi = doi_elem.text.strip() if doi_elem is not None and doi_elem.text else ""
        citation["DOI"] = doi
        if doi:
            cited_dois.append(doi)
        
        # Year
        date_elem = bibl.find(".//tei:monogr/tei:imprint/tei:date[@type='published']", NS)
        citation["year"] = date_elem.get("when", "")[:4] if date_elem is not None else ""
        
        citations.append(citation)
    
    metadata["citations"] = citations
    metadata["n_citations"] = len(citations)
    metadata["cited_dois"] = cited_dois
    metadata["cited_authors"] = list(set(cited_authors))  # Unique authors
    
    return metadata


def extract_plain_text_from_tei(tei_xml):
    """
    Extract plain text from TEI XML body.
    
    Args:
        tei_xml: TEI XML string from Grobid
        
    Returns:
        str: Plain text content
    """
    root = ET.fromstring(tei_xml.encode('utf-8'))
    
    # Find the body
    body = root.find(".//tei:text/tei:body", NS)
    if body is None:
        return ""
    
    # Remove references and figures
    for ref in body.findall(".//tei:ref", NS):
        if ref.text:
            ref.text = ""
        ref.tail = ref.tail or ""
    
    for figure in body.findall(".//tei:figure", NS):
        figure.getparent().remove(figure)
    
    # Extract text from all paragraphs
    paragraphs = []
    for p in body.findall(".//tei:p", NS):
        text = "".join(p.itertext()).strip()
        text = re.sub(r'\s+', ' ', text)
        if text:
            paragraphs.append(text)
    
    return "\n\n".join(paragraphs)


def build_corpus_metadata_and_table(project_root, metadata_filename="corpus_metadata.json"):
    """
    Build corpus metadata and table from processed PDFs.
    
    Args:
        project_root: Root directory containing P#### folders
        metadata_filename: Name for metadata JSON file
        
    Returns:
        pd.DataFrame: Corpus table with all metadata
    """
    project_path = Path(project_root)
    
    # Scan P#### folders and collect metadata
    metadata_list = []
    for paper_dir in sorted(project_path.glob("P*")):
        if not paper_dir.is_dir():
            continue
            
        metadata_file = paper_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                meta = json.load(f)
                metadata_list.append(meta)
    
    # Save collected metadata to corpus_metadata.json
    metadata_path = project_path / metadata_filename
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)
    
    print(f"✔ Wrote {metadata_filename} with {len(metadata_list)} papers")
    
    # Create DataFrame
    rows = []
    for meta in metadata_list:
        # Handle authors - could be list of dicts or list of strings
        authors = meta.get("authors", [])
        if authors and isinstance(authors[0], dict):
            authors_str = "; ".join([a.get("full_name", "") for a in authors])
        else:
            authors_str = "; ".join(authors) if isinstance(authors, list) else str(authors)
        
        # Extract filename from pdf_path
        pdf_path = meta.get("pdf_path", "") or meta.get("filename", "")
        pdf_filename = Path(pdf_path).name if pdf_path else ""
        
        row = {
            "paper_id": meta.get("paper_id", ""),
            "title": meta.get("title", ""),
            "authors": authors_str,
            "n_authors": len(authors) if isinstance(authors, list) else 0,
            "abstract": meta.get("abstract", ""),
            "keywords": "; ".join(meta.get("keywords", [])) if isinstance(meta.get("keywords"), list) else "",
            "DOI": meta.get("DOI", "") or meta.get("doi", ""),
            "journal": meta.get("journal", ""),
            "publication_date": meta.get("publication_date", "") or str(meta.get("year", "")),
            "pdf_path": pdf_path,
            "pdf_filename": pdf_filename,
            "n_citations": meta.get("n_citations", 0) or len(meta.get("citations", [])),
            "citations": json.dumps(meta.get("citations", [])),
            # Extract cited titles (all citations have titles)
            "cited_titles": "; ".join([c.get("title", "").strip() for c in meta.get("citations", []) if c.get("title", "").strip()]),
            # Extract cited DOIs (only when available, semicolon-separated)
            "cited_dois": "; ".join([c.get("doi", "") or c.get("DOI", "") for c in meta.get("citations", []) if c.get("doi") or c.get("DOI")]),
            # Extract all cited authors (from all citations, not just those with DOIs)
            "cited_authors": "; ".join([
                author 
                for c in meta.get("citations", [])
                for author in (c.get("authors", []) if isinstance(c.get("authors"), list) else [])
            ])
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save to CSV and JSONL
    csv_path = project_path / "corpus_table.csv"
    jsonl_path = project_path / "corpus_table.jsonl"
    
    df.to_csv(csv_path, index=False)
    df.to_json(jsonl_path, orient='records', lines=True)
    
    print(f"✔ Wrote corpus_table.csv with {len(df)} rows")
    print(f"✔ Wrote corpus_table.jsonl")
    
    return df


def run_llm(
    messages,
    client,
    model="gpt-oss-120b",
    temperature=0.2,
    top_p=0.9,
    max_tokens=16384,
    retries=2
):
    """
    Safe LLM call wrapper with retry logic.
    
    Args:
        messages: List of message dicts for OpenAI API
        client: OpenAI client instance
        model: Model name to use
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        max_tokens: Maximum tokens in response
        retries: Number of retry attempts on failure
        
    Returns:
        str: LLM response text
        
    Raises:
        RuntimeError: If all retry attempts fail
    """
    last_exc = None

    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                messages=messages,
            )
            return response.choices[0].message.content

        except Exception as e:
            last_exc = e
            if attempt < retries:
                print(f"⚠️ LLM error, retrying ({attempt+1}/{retries})…")
            else:
                raise RuntimeError(
                    f"LLM failed after {retries+1} attempts: {e}"
                )

    raise last_exc


def _build_messages(title, description):
    """
    Create OpenAI-compatible messages with system prompt for topic extraction.
    
    Args:
        title: Book title
        description: Book description
        
    Returns:
        list: Message dicts for OpenAI API
    """
    user_content = f"TITLE: {title.strip()}\n\nDESCRIPTION:\n{description.strip()}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _sanitize_and_validate_topics(raw):
    """
    Convert raw LLM output into validated topic structure.
    
    Args:
        raw: Raw LLM response string
        
    Returns:
        dict: Validated topics dict with schema:
              {"topics": [{"category": str, "label": str, "confidence": float, "rationale": str}, ...]}
              
    Raises:
        ValueError: If output doesn't match expected schema
    """
    raw = raw.strip()

    # Remove accidental markdown fences like ```json ... ```
    if raw.startswith("```"):
        raw = raw.strip("` \n")
        if raw.lower().startswith("json"):
            raw = raw[4:].lstrip()

    # Try strict JSON; fallback to extracting text between first { and last }
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        first = raw.find("{")
        last = raw.rfind("}")
        if first != -1 and last != -1 and last > first:
            obj = json.loads(raw[first:last+1])
        else:
            raise

    # Minimal schema check
    if not isinstance(obj, dict) or "topics" not in obj or not isinstance(obj["topics"], list):
        raise ValueError("Model output missing top-level 'topics' list.")

    cleaned = []
    for t in obj["topics"]:
        if not isinstance(t, dict):
            continue
        category = (t.get("category") or "").strip()
        label = (t.get("label") or "").strip()
        rat = (t.get("rationale") or "").strip()
        try:
            conf = float(t.get("confidence", 0.0))
        except Exception:
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        if label:
            # Use "General" as fallback if no category provided
            if not category:
                category = "General"
            cleaned.append({
                "category": category,
                "label": label, 
                "confidence": conf, 
                "rationale": rat
            })

    # Enforce 3–8 topics if possible; clip if too many
    if len(cleaned) > 8:
        cleaned = cleaned[:8]

    return {"topics": cleaned}


def add_topics_to_corpus_table(output_root, client, inplace=False, model="gpt-oss-120b", 
                               progress_callback=None, status_callback=None):
    """
    Read corpus_table.jsonl, call LLM for each record's description, and add topics.
    
    Args:
        output_root: Path to project directory containing corpus_table.jsonl
        client: OpenAI client instance
        inplace: If True, replace original file (with backup); if False, create new file
        model: Model name to use
        progress_callback: Optional function(current, total) to report progress
        status_callback: Optional function(message) to report status text
        
    Returns:
        None (writes output file)
    """
    from datetime import datetime, timezone
    
    output_root = Path(output_root)
    inp = output_root / "corpus_table.jsonl"
    if not inp.exists():
        raise FileNotFoundError(f"Missing input file: {inp}")

    tmp_out = output_root / "corpus_table.with_topics.tmp.jsonl"
    final_out = output_root / "corpus_table.with_topics.jsonl"

    processed, skipped = 0, 0
    
    # First pass: count total records
    total_records = sum(1 for _ in inp.open("r", encoding="utf-8"))
    current = 0

    with inp.open("r", encoding="utf-8") as fin, tmp_out.open("w", encoding="utf-8") as fout:
        for line in fin:
            current += 1
            rec = json.loads(line)
            title = rec.get("title", "")
            description = (rec.get("description") or "").strip()
            book_id = rec.get("book_id", "unknown")

            if description:
                # Update status
                if status_callback:
                    status_callback(f"🤖 Processing {current}/{total_records}: {book_id} - {title[:50]}...")
                
                # Direct LLM call
                messages = _build_messages(title, description)
                raw = run_llm(messages=messages, client=client, model=model, temperature=0.2, top_p=0.9)
                topics_obj = _sanitize_and_validate_topics(raw)

                rec["topics"] = topics_obj["topics"]
                rec["topics_ts"] = datetime.now(timezone.utc).isoformat()
                processed += 1
            else:
                # No description: keep record and mark it
                if status_callback:
                    status_callback(f"⏭️ Skipping {current}/{total_records}: {book_id} (no description)")
                rec["topics"] = []
                rec["topics_note"] = "no_description"
                skipped += 1

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
            # Update progress
            if progress_callback:
                progress_callback(current, total_records)

    # Atomic finalize
    tmp_out.replace(final_out)

    if inplace:
        # Keep a backup of the original and swap
        backup = output_root / "corpus_table.backup.jsonl"
        inp.replace(backup)
        final_out.replace(inp)

    print("---------------------------------------------------")
    print("Topics enrichment completed.")
    print(f"Processed (with description): {processed}")
    print(f"Skipped (no description):     {skipped}")
    print(f"Output file:                  {inp if inplace else final_out}")
    print("---------------------------------------------------")


def build_paper_ego_graph(paper_record, all_records=None, include_topics=True, 
                         include_authors=True, include_cited_papers=True,
                         include_cited_authors=False):
    """
    Build an ego graph centered on a single book.
    
    The ego graph shows the book's immediate network neighborhood:
    - The book itself (center node)
    - Its author (if include_authors=True)
    - Its topics (if include_topics=True)
    - Recommended books (if include_cited_papers=True)
    
    Parameters
    ----------
    paper_record : dict
        Single book record from corpus_table.with_topics.jsonl
    all_records : list of dict, optional
        All book records (needed to find recommended books within corpus)
    include_topics : bool
        Add topic nodes connected to the book
    include_authors : bool
        Add author node connected to the book
    include_cited_papers : bool
        Add recommended books as nodes
    include_cited_authors : bool
        Not used for books (kept for compatibility)
    
    Returns
    -------
    tuple (networkx.DiGraph, pyvis.network.Network)
        - nx_graph: NetworkX directed graph
        - net: Pyvis Network ready for visualization
    """
    try:
        import networkx as nx
        from pyvis.network import Network
    except ImportError as e:
        raise ImportError(
            "Please install required packages: pip install networkx pyvis"
        ) from e
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Extract book info
    book_id = paper_record.get("book_id", "unknown")
    title = paper_record.get("title", "Untitled")
    
    # Use book_id as node ID
    node_id = f"book:{book_id}"
    
    # Add center book node
    G.add_node(
        node_id,
        label=title[:30] + "..." if len(title) > 30 else title,
        title=f"{title}\nID: {book_id}",
        node_type="book",
        color="#4A90E2",  # Blue for book
        size=30,
        shape="dot"
    )
    
    # Add author
    if include_authors:
        author_name = paper_record.get("author", "Unknown")
        
        if author_name and author_name != "Unknown":
            author_id = f"author:{author_name}"
            G.add_node(
                author_id,
                label=author_name,
                title=f"Author: {author_name}",
                node_type="author",
                color="#50C878",  # Green for authors
                size=20,
                shape="triangle"
            )
            # Edge from book to author
            G.add_edge(node_id, author_id, label="written_by")
    
    # Add topics (both categories and specific topics)
    if include_topics:
        topics = paper_record.get("topics", [])
        
        # Track categories to avoid duplicates
        added_categories = set()
        
        for topic in topics:
            category = topic.get("category", "General")
            topic_label = topic.get("label", "unknown topic")
            confidence = topic.get("confidence", 0.5)
            
            # Add category node (broad topic) - only once per unique category
            category_id = f"category:{category}"
            if category_id not in added_categories:
                added_categories.add(category_id)
                if category_id not in G.nodes:
                    G.add_node(
                        category_id,
                        label=category,
                        title=f"Category: {category}",
                        node_type="category",
                        color="#9B59B6",  # Purple for categories
                        size=25,  # Larger for categories
                        shape="box"
                    )
                # Connect paper to category
                G.add_edge(node_id, category_id, label="in_category")
            
            # Add specific topic node
            topic_id = f"topic:{topic_label}"
            if topic_id not in G.nodes:
                G.add_node(
                    topic_id,
                    label=topic_label,
                    title=f"Topic: {topic_label}\nCategory: {category}\nConfidence: {confidence:.2f}",
                    node_type="topic",
                    color="#FF6B6B",  # Red for specific topics
                    size=10 + confidence * 15,  # Size by confidence
                    shape="ellipse"
                )
            
            # Connect paper to specific topic
            G.add_edge(node_id, topic_id, label="has_topic", weight=confidence)
            
            # Connect specific topic to its category
            G.add_edge(topic_id, category_id, label="belongs_to", style="dashed")
    
    # Add recommended books
    if include_cited_papers:
        # Build lookup of titles to records in our corpus (lowercase for matching)
        title_to_record = {}
        if all_records:
            for rec in all_records:
                rec_title = rec.get("title", "").strip().lower()
                if rec_title:
                    title_to_record[rec_title] = rec
        
        # Get recommended books (stored as list)
        recommended_books = paper_record.get("recommended_books", [])
        if not isinstance(recommended_books, list):
            recommended_books = []
        
        # Add all recommended books (whether in our corpus or not)
        for rec_book_title in recommended_books:
            rec_book_title_clean = rec_book_title.strip()
            rec_book_title_lower = rec_book_title_clean.lower()
            rec_node_id = f"recommended:{rec_book_title_clean[:50]}"  # Use title prefix as node ID
            
            # Check if recommended book is in our corpus
            if rec_book_title_lower in title_to_record:
                rec_book_rec = title_to_record[rec_book_title_lower]
                rec_book_id = rec_book_rec.get("book_id", "unknown")
                node_color = "#9B59B6"  # Purple for books in corpus
                node_label = rec_book_title_clean[:25] + "..." if len(rec_book_title_clean) > 25 else rec_book_title_clean
                node_title_text = f"{rec_book_title_clean}\n(in corpus: {rec_book_id})"
            else:
                # Recommended book NOT in our corpus
                node_color = "#FFA500"  # Orange for external books
                # Truncate long titles for label
                node_label = rec_book_title_clean[:25] + "..." if len(rec_book_title_clean) > 25 else rec_book_title_clean
                node_title_text = f"{rec_book_title_clean}\n(not in corpus)"
            
            # Add recommended book node if not already present
            if rec_node_id not in G.nodes:
                G.add_node(
                    rec_node_id,
                    label=node_label,
                    title=node_title_text,
                    node_type="recommended_book",
                    color=node_color,
                    size=15,
                    shape="dot"
                )
            
            # Add recommendation edge
            G.add_edge(node_id, rec_node_id, label="recommends", color="#999999")
    
    # Note: include_cited_authors not used for books, kept for compatibility
    
    # Create Pyvis visualization
    net = Network(
        height="600px",
        width="100%",
        bgcolor="#ffffff",
        font_color="black",
        directed=True
    )
    
    # Import from NetworkX
    net.from_nx(G)
    
    # Configure physics for better layout
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "stabilization": {
          "iterations": 200
        },
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 150,
          "springConstant": 0.04
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100
      }
    }
    """)
    
    return G, net


def build_full_corpus_graph(all_records, include_topics=True, include_authors=True, 
                           include_cited_papers=False, include_cited_authors=False, 
                           min_topic_confidence=0.0):
    """
    Build a complete knowledge graph from all books in the corpus.
    
    Shows the full network including:
    - All books as nodes
    - All authors (with connections to their books)
    - All topics (showing thematic clusters)
    - Recommended books
    
    Parameters
    ----------
    all_records : list of dict
        All book records from corpus_table.with_topics.jsonl
    include_topics : bool
        Add topic nodes connected to books
    include_authors : bool
        Add author nodes connected to books
    include_cited_papers : bool
        Add recommended books as nodes
    include_cited_authors : bool
        Not used for books (kept for compatibility)
    min_topic_confidence : float
        Minimum confidence score to include a topic (0.0 to 1.0)
    
    Returns
    -------
    tuple (networkx.DiGraph, pyvis.network.Network)
        - nx_graph: NetworkX directed graph
        - net: Pyvis Network ready for visualization
    """
    try:
        import networkx as nx
        from pyvis.network import Network
    except ImportError as e:
        raise ImportError(
            "Please install required packages: pip install networkx pyvis"
        ) from e
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Track all unique authors, topics, and categories across corpus
    all_authors = {}     # author_name -> list of node_ids
    all_topics = {}      # topic_label -> list of (node_id, confidence, category)
    all_categories = {}  # category_name -> list of node_ids (books in that category)
    
    # First pass: Add all books and collect authors/topics
    for rec in all_records:
        book_id = rec.get("book_id", "unknown")
        title = rec.get("title", "Untitled")
        rating_count = rec.get("rating_count", 0)
        average_rating = rec.get("average_rating", 0)
        
        # Use book_id as node ID
        node_id = f"book:{book_id}"
        
        # Add book node (size by rating count)
        G.add_node(
            node_id,
            label=title[:25] + "..." if len(title) > 25 else title,
            title=f"{title}\nID: {book_id}\nRating: {average_rating}\nRatings: {rating_count}",
            node_type="book",
            color="#4A90E2",  # Blue for books
            size=15 + min(rating_count // 100, 30),  # Size by rating count
            shape="dot"
        )
        
        # Collect author
        if include_authors:
            author_name = rec.get("author", "")
            if author_name and author_name != "Unknown":
                if author_name not in all_authors:
                    all_authors[author_name] = []
                all_authors[author_name].append(node_id)
        
        # Collect topics (both categories and specific topics)
        if include_topics:
            topics = rec.get("topics", [])
            for topic in topics:
                category = topic.get("category", "General")
                topic_label = topic.get("label", "unknown topic")
                confidence = topic.get("confidence", 0.5)
                
                if confidence >= min_topic_confidence:
                    # Track categories
                    if category not in all_categories:
                        all_categories[category] = []
                    all_categories[category].append(node_id)
                    
                    # Track specific topics
                    if topic_label not in all_topics:
                        all_topics[topic_label] = []
                    all_topics[topic_label].append((node_id, confidence, category))
    
    # Second pass: Add author nodes and edges
    if include_authors:
        for author_name, node_ids in all_authors.items():
            # Author node size by number of books
            author_id = f"author:{author_name}"
            G.add_node(
                author_id,
                label=author_name,
                title=f"Author: {author_name}\nBooks: {len(node_ids)}",
                node_type="author",
                color="#50C878",  # Green for authors
                size=10 + len(node_ids) * 5,  # Size by productivity
                shape="triangle"
            )
            
            # Connect author to all their books
            for node_id in node_ids:
                G.add_edge(author_id, node_id, label="wrote")
    
    # Third pass: Add category nodes (broad topics) and topic nodes (specific topics)
    if include_topics:
        # First add category nodes
        for category_name, book_ids in all_categories.items():
            category_id = f"category:{category_name}"
            G.add_node(
                category_id,
                label=category_name,
                title=f"Category: {category_name}\nBooks: {len(book_ids)}",
                node_type="category",
                color="#9B59B6",  # Purple for categories
                size=20 + len(book_ids) * 4,  # Larger size for categories
                shape="box"
            )
            
            # Connect category to books in it
            for book_id in book_ids:
                G.add_edge(book_id, category_id, label="in_category")
        
        # Then add specific topic nodes
        for topic_label, topic_data in all_topics.items():
            topic_id = f"topic:{topic_label}"
            avg_confidence = sum(conf for _, conf, _ in topic_data) / len(topic_data)
            
            # Get category (should be same for all instances of this topic)
            category = topic_data[0][2] if topic_data else "General"
            category_id = f"category:{category}"
            
            G.add_node(
                topic_id,
                label=topic_label,
                title=f"Topic: {topic_label}\nCategory: {category}\nBooks: {len(topic_data)}\nAvg confidence: {avg_confidence:.2f}",
                node_type="topic",
                color="#FF6B6B",  # Red for specific topics
                size=10 + len(topic_data) * 2,  # Size by prevalence
                shape="ellipse"
            )
            
            # Connect topic to books
            for book_id, confidence, _ in topic_data:
                G.add_edge(book_id, topic_id, label="has_topic", weight=confidence)
            
            # Connect topic to its category
            if category_id in G.nodes:
                G.add_edge(topic_id, category_id, label="belongs_to", style="dashed")
    
    # Fourth pass: Add recommended books
    if include_cited_papers:
        # Build title-to-record lookup for matching recommended books to corpus
        title_to_record = {}
        for rec in all_records:
            rec_title = rec.get("title", "").strip().lower()
            if rec_title:
                title_to_record[rec_title] = rec
        
        for rec in all_records:
            book_id = rec.get("book_id", "unknown")
            recommending_node_id = f"book:{book_id}"
            
            # Get recommended books (stored as list)
            recommended_books = rec.get("recommended_books", [])
            if not isinstance(recommended_books, list):
                recommended_books = []
            
            # Add recommended books (whether in corpus or not)
            for rec_book_title in recommended_books:
                rec_book_title_clean = rec_book_title.strip()
                rec_book_title_lower = rec_book_title_clean.lower()
                
                # Check if recommended book is in our corpus (by title)
                if rec_book_title_lower in title_to_record:
                    # Book IS in corpus - get its node_id
                    rec_book_rec = title_to_record[rec_book_title_lower]
                    rec_corpus_book_id = rec_book_rec.get("book_id", "unknown")
                    rec_corpus_node_id = f"book:{rec_corpus_book_id}"
                    # Add edge to existing corpus node
                    if rec_corpus_node_id in G.nodes:
                        G.add_edge(recommending_node_id, rec_corpus_node_id, label="recommends", color="#999999")
                else:
                    # Book NOT in corpus - add as external node
                    rec_node_id = f"recommended:{rec_book_title_clean[:50]}"  # Use title prefix as node ID
                    
                    if rec_node_id not in G.nodes:
                        # Truncate long titles for label
                        node_label = rec_book_title_clean[:25] + "..." if len(rec_book_title_clean) > 25 else rec_book_title_clean
                        node_title_text = f"{rec_book_title_clean}\n(not in corpus)"
                        
                        G.add_node(
                            rec_node_id,
                            label=node_label,
                            title=node_title_text,
                            node_type="recommended_book",
                            color="#FFA500",  # Orange for external books
                            size=10,
                            shape="dot"
                        )
                    # Add recommendation edge
                    G.add_edge(recommending_node_id, rec_node_id, label="recommends", color="#999999")
    
    # Note: include_cited_authors not used for books, kept for compatibility
    
    # Create Pyvis visualization
    net = Network(
        height="800px",
        width="100%",
        bgcolor="#ffffff",
        font_color="black",
        directed=True
    )
    
    # Import from NetworkX
    net.from_nx(G)
    
    # Configure physics for larger graph
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "stabilization": {
          "iterations": 300
        },
        "barnesHut": {
          "gravitationalConstant": -15000,
          "centralGravity": 0.1,
          "springLength": 200,
          "springConstant": 0.02,
          "damping": 0.5
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true,
        "keyboard": true
      }
    }
    """)
    
    return G, net
