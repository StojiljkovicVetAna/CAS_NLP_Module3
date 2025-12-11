#!/usr/bin/env python3
"""
MCP Server for Goodreads Book Knowledge Graph

This server provides tools for Claude Desktop to interact with the book corpus:
- Search books by title or author
- Get book details and metadata
- Extract topics from book descriptions
- Generate knowledge graphs
- Add new books to the corpus
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource

# Import your utilities
from utils_KG import (
    add_topics_to_corpus_table,
    build_paper_ego_graph,
    build_full_corpus_graph,
    _build_messages,
    _sanitize_and_validate_topics,
    run_llm
)

from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
BASE_URL = os.getenv("BASE_URL", "")
GPUSTACK_API_KEY = os.getenv("GPUSTACK_API_KEY", "")

# Initialize OpenAI client
client = OpenAI(
    base_url=BASE_URL,
    api_key=GPUSTACK_API_KEY
)

# Default corpus path
DEFAULT_CORPUS_PATH = Path("/storage/homefs/as23z124/NLP_CAS_M3_project/goodreads_corpus")
# Original CSV path
ORIGINAL_CSV_PATH = Path("/storage/homefs/as23z124/NLP_CAS_M3_project/goodreads_books.csv")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-books-server")

# Create the MCP server
app = Server("goodreads-knowledge-graph")


def load_corpus(corpus_path: Path) -> List[Dict[str, Any]]:
    """Load all books from the corpus."""
    corpus_file = corpus_path / "corpus_table.with_topics.jsonl"
    
    # Fall back to corpus_table.jsonl if topics file doesn't exist
    if not corpus_file.exists():
        corpus_file = corpus_path / "corpus_table.jsonl"
    
    if not corpus_file.exists():
        return []
    
    books = []
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            books.append(json.loads(line))
    
    return books


def save_book_to_corpus(book: Dict[str, Any], corpus_path: Path):
    """Save a single book to the corpus."""
    corpus_file = corpus_path / "corpus_table.jsonl"
    
    with open(corpus_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(book, ensure_ascii=False) + '\n')
    
    logger.info(f"Added book {book.get('book_id')} to corpus")


def load_original_csv() -> List[Dict[str, Any]]:
    """Load the original Goodreads CSV file."""
    import pandas as pd
    
    if not ORIGINAL_CSV_PATH.exists():
        logger.warning(f"Original CSV not found: {ORIGINAL_CSV_PATH}")
        return []
    
    try:
        df = pd.read_csv(ORIGINAL_CSV_PATH)
        return df.to_dict('records')
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        return []


def find_book_in_csv(book_id: str, csv_books: List[Dict]) -> Optional[Dict]:
    """Find a book by ID in the CSV data."""
    for book in csv_books:
        if str(book.get('id', '')) == str(book_id):
            return book
    return None


def csv_to_corpus_format(csv_row: Dict) -> Dict[str, Any]:
    """Convert a CSV row to corpus format."""
    # Parse recommended books
    recommended_books_raw = str(csv_row.get('recommended_books', ''))
    recommended_books = []
    if recommended_books_raw and recommended_books_raw != 'nan':
        books_list = [b.strip() for b in recommended_books_raw.replace(';', ',').split(',') if b.strip()]
        recommended_books = books_list
    
    # Parse books in series
    books_in_series_raw = str(csv_row.get('books_in_series', ''))
    books_in_series = []
    if books_in_series_raw and books_in_series_raw != 'nan':
        books_in_series = [b.strip() for b in books_in_series_raw.replace(';', ',').split(',') if b.strip()]
    
    return {
        "book_id": str(csv_row.get('id', '')),
        "title": str(csv_row.get('title', 'Untitled')),
        "author": str(csv_row.get('author', 'Unknown')),
        "description": str(csv_row.get('description', '')),
        "link": str(csv_row.get('link', '')),
        "cover_link": str(csv_row.get('cover_link', '')),
        "author_link": str(csv_row.get('author_link', '')),
        "publisher": str(csv_row.get('publisher', '')),
        "date_published": str(csv_row.get('date_published', '')),
        "original_title": str(csv_row.get('original_title', '')),
        "genre_and_votes": str(csv_row.get('genre_and_votes', '')),
        "series": str(csv_row.get('series', '')),
        "number_of_pages": csv_row.get('number_of_pages', ''),
        "isbn": str(csv_row.get('isbn', '')),
        "isbn13": str(csv_row.get('isbn13', '')),
        "average_rating": csv_row.get('average_rating', 0),
        "rating_count": csv_row.get('rating_count', 0),
        "review_count": csv_row.get('review_count', 0),
        "settings": str(csv_row.get('settings', '')),
        "characters": str(csv_row.get('characters', '')),
        "awards": str(csv_row.get('awards', '')),
        "recommended_books": recommended_books,
        "books_in_series": books_in_series,
    }


def search_books(query: str, books: List[Dict], search_type: str = "all") -> List[Dict]:
    """
    Search books by title, author, or both.
    
    Args:
        query: Search query
        books: List of book records
        search_type: 'title', 'author', or 'all'
    """
    query_lower = query.lower()
    results = []
    
    for book in books:
        if search_type in ["title", "all"]:
            title = book.get("title", "").lower()
            if query_lower in title:
                results.append(book)
                continue
        
        if search_type in ["author", "all"]:
            author = book.get("author", "").lower()
            if query_lower in author:
                results.append(book)
    
    return results


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools for the MCP server."""
    return [
        Tool(
            name="search_books",
            description="Search for books in the corpus by title or author. Returns matching books with their metadata.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (book title or author name)"
                    },
                    "search_type": {
                        "type": "string",
                        "enum": ["title", "author", "all"],
                        "description": "Type of search: 'title', 'author', or 'all' (default: 'all')",
                        "default": "all"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 10)",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_book_details",
            description="Get detailed information about a specific book by its ID, including topics, author, description, and recommended books.",
            inputSchema={
                "type": "object",
                "properties": {
                    "book_id": {
                        "type": "string",
                        "description": "The unique book ID"
                    }
                },
                "required": ["book_id"]
            }
        ),
        Tool(
            name="extract_book_topics",
            description="Extract topics from a book's description using LLM analysis. Returns hierarchical topics with categories, labels, confidence scores, and rationales.",
            inputSchema={
                "type": "object",
                "properties": {
                    "book_id": {
                        "type": "string",
                        "description": "Book ID to extract topics for"
                    },
                    "title": {
                        "type": "string",
                        "description": "Book title (required if book_id is not in corpus)"
                    },
                    "description": {
                        "type": "string",
                        "description": "Book description (required if book_id is not in corpus)"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="generate_ego_graph",
            description="Generate a knowledge graph visualization centered on a specific book, showing its author, topics, and recommended books. Returns HTML file path.",
            inputSchema={
                "type": "object",
                "properties": {
                    "book_id": {
                        "type": "string",
                        "description": "The book ID to center the ego graph on"
                    },
                    "include_topics": {
                        "type": "boolean",
                        "description": "Include topic nodes (default: true)",
                        "default": True
                    },
                    "include_authors": {
                        "type": "boolean",
                        "description": "Include author nodes (default: true)",
                        "default": True
                    },
                    "include_recommended": {
                        "type": "boolean",
                        "description": "Include recommended books (default: true)",
                        "default": True
                    }
                },
                "required": ["book_id"]
            }
        ),
        Tool(
            name="add_book_to_corpus",
            description="Add a new book to the corpus with metadata. Optionally extracts topics automatically.",
            inputSchema={
                "type": "object",
                "properties": {
                    "book_id": {
                        "type": "string",
                        "description": "Unique identifier for the book"
                    },
                    "title": {
                        "type": "string",
                        "description": "Book title"
                    },
                    "author": {
                        "type": "string",
                        "description": "Author name"
                    },
                    "description": {
                        "type": "string",
                        "description": "Book description or summary"
                    },
                    "recommended_books": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of recommended book titles",
                        "default": []
                    },
                    "extract_topics": {
                        "type": "boolean",
                        "description": "Automatically extract topics using LLM (default: true)",
                        "default": True
                    }
                },
                "required": ["book_id", "title", "author", "description"]
            }
        ),
        Tool(
            name="get_books_by_author",
            description="Get all books by a specific author in the corpus.",
            inputSchema={
                "type": "object",
                "properties": {
                    "author_name": {
                        "type": "string",
                        "description": "Author name to search for"
                    }
                },
                "required": ["author_name"]
            }
        ),
        Tool(
            name="get_books_by_topic",
            description="Get all books that have a specific topic or category.",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic_query": {
                        "type": "string",
                        "description": "Topic label or category to search for"
                    },
                    "min_confidence": {
                        "type": "number",
                        "description": "Minimum confidence score (0.0 to 1.0, default: 0.0)",
                        "default": 0.0
                    }
                },
                "required": ["topic_query"]
            }
        ),
        Tool(
            name="get_corpus_stats",
            description="Get statistics about the corpus: total books, authors, topics, etc.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="import_book_from_csv",
            description="Import a book from the original Goodreads CSV into the corpus with topic extraction. Use this when a book exists in the CSV but not yet in the corpus.",
            inputSchema={
                "type": "object",
                "properties": {
                    "book_id": {
                        "type": "string",
                        "description": "The book ID from the original CSV to import"
                    },
                    "extract_topics": {
                        "type": "boolean",
                        "description": "Automatically extract topics using LLM (default: true)",
                        "default": True
                    }
                },
                "required": ["book_id"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls from Claude."""
    
    try:
        # Load corpus
        books = load_corpus(DEFAULT_CORPUS_PATH)
        
        if name == "search_books":
            query = arguments["query"]
            search_type = arguments.get("search_type", "all")
            limit = arguments.get("limit", 10)
            
            results = search_books(query, books, search_type)[:limit]
            
            if not results:
                return [TextContent(
                    type="text",
                    text=f"No books found matching '{query}'"
                )]
            
            # Format results
            output = f"Found {len(results)} book(s) matching '{query}':\n\n"
            for book in results:
                output += f"**{book.get('title', 'Untitled')}**\n"
                output += f"- ID: {book.get('book_id', 'N/A')}\n"
                output += f"- Author: {book.get('author', 'Unknown')}\n"
                output += f"- Rating: {book.get('average_rating', 'N/A')} ({book.get('rating_count', 0)} ratings)\n"
                desc = book.get('description', '')
                if desc:
                    output += f"- Description: {desc[:200]}...\n"
                output += "\n"
            
            return [TextContent(type="text", text=output)]
        
        elif name == "get_book_details":
            book_id = arguments["book_id"]
            
            # Find book
            book = None
            for b in books:
                if b.get("book_id") == book_id:
                    book = b
                    break
            
            if not book:
                return [TextContent(
                    type="text",
                    text=f"Book with ID '{book_id}' not found in corpus."
                )]
            
            # Format book details
            output = f"**{book.get('title', 'Untitled')}**\n\n"
            output += f"**Author:** {book.get('author', 'Unknown')}\n"
            output += f"**ID:** {book.get('book_id')}\n"
            output += f"**Rating:** {book.get('average_rating', 'N/A')} ({book.get('rating_count', 0)} ratings)\n"
            output += f"**Publisher:** {book.get('publisher', 'N/A')}\n"
            output += f"**Published:** {book.get('date_published', 'N/A')}\n"
            
            if book.get('series'):
                output += f"**Series:** {book.get('series')}\n"
            
            output += f"\n**Description:**\n{book.get('description', 'No description available.')}\n"
            
            # Topics
            topics = book.get('topics', [])
            if topics:
                output += f"\n**Topics ({len(topics)}):**\n"
                for topic in topics:
                    output += f"- **{topic.get('label')}** ({topic.get('category')}) - Confidence: {topic.get('confidence', 0):.2f}\n"
                    output += f"  _{topic.get('rationale', '')}_\n"
            
            # Recommended books
            recommended = book.get('recommended_books', [])
            if recommended:
                output += f"\n**Recommended Books ({len(recommended)}):**\n"
                for rec in recommended[:5]:
                    output += f"- {rec}\n"
            
            return [TextContent(type="text", text=output)]
        
        elif name == "extract_book_topics":
            book_id = arguments.get("book_id")
            title = arguments.get("title")
            description = arguments.get("description")
            
            # Try to get from corpus first
            if book_id:
                for b in books:
                    if b.get("book_id") == book_id:
                        title = b.get("title", "Untitled")
                        description = b.get("description", "")
                        break
            
            if not title or not description:
                return [TextContent(
                    type="text",
                    text="Error: Need either book_id or both title and description."
                )]
            
            # Extract topics using LLM
            messages = _build_messages(title, description)
            raw = run_llm(messages=messages, client=client, model="gpt-oss-120b")
            topics_obj = _sanitize_and_validate_topics(raw)
            
            # Format output
            output = f"**Extracted topics for '{title}':**\n\n"
            for topic in topics_obj["topics"]:
                output += f"**{topic['label']}** ({topic['category']})\n"
                output += f"- Confidence: {topic['confidence']:.2f}\n"
                output += f"- Rationale: {topic['rationale']}\n\n"
            
            return [TextContent(type="text", text=output)]
        
        elif name == "generate_ego_graph":
            book_id = arguments["book_id"]
            include_topics = arguments.get("include_topics", True)
            include_authors = arguments.get("include_authors", True)
            include_recommended = arguments.get("include_recommended", True)
            
            # Find book
            book = None
            for b in books:
                if b.get("book_id") == book_id:
                    book = b
                    break
            
            if not book:
                return [TextContent(
                    type="text",
                    text=f"Book with ID '{book_id}' not found in corpus."
                )]
            
            # Generate ego graph
            nx_graph, pyvis_net = build_paper_ego_graph(
                book,
                all_records=books,
                include_topics=include_topics,
                include_authors=include_authors,
                include_cited_papers=include_recommended
            )
            
            # Save HTML
            output_path = DEFAULT_CORPUS_PATH / f"{book_id}_ego_graph.html"
            pyvis_net.save_graph(str(output_path))
            
            title = book.get("title", "Untitled")
            return [TextContent(
                type="text",
                text=f"Generated ego graph for '{title}' (ID: {book_id})\n\n"
                     f"Graph saved to: {output_path}\n"
                     f"Nodes: {nx_graph.number_of_nodes()}\n"
                     f"Edges: {nx_graph.number_of_edges()}"
            )]
        
        elif name == "add_book_to_corpus":
            book_id = arguments["book_id"]
            title = arguments["title"]
            author = arguments["author"]
            description = arguments["description"]
            recommended_books = arguments.get("recommended_books", [])
            extract_topics = arguments.get("extract_topics", True)
            
            # Check if book already exists
            for b in books:
                if b.get("book_id") == book_id:
                    return [TextContent(
                        type="text",
                        text=f"Book with ID '{book_id}' already exists in corpus."
                    )]
            
            # Create book record
            book = {
                "book_id": book_id,
                "title": title,
                "author": author,
                "description": description,
                "recommended_books": recommended_books,
                "link": "",
                "cover_link": "",
                "author_link": "",
                "publisher": "",
                "date_published": "",
                "original_title": "",
                "genre_and_votes": "",
                "series": "",
                "number_of_pages": "",
                "isbn": "",
                "isbn13": "",
                "average_rating": 0,
                "rating_count": 0,
                "review_count": 0,
                "settings": "",
                "characters": "",
                "awards": "",
                "books_in_series": []
            }
            
            # Extract topics if requested
            if extract_topics and description:
                try:
                    messages = _build_messages(title, description)
                    raw = run_llm(messages=messages, client=client, model="gpt-oss-120b")
                    topics_obj = _sanitize_and_validate_topics(raw)
                    book["topics"] = topics_obj["topics"]
                    book["topics_ts"] = datetime.now(timezone.utc).isoformat()
                except Exception as e:
                    logger.error(f"Error extracting topics: {e}")
                    book["topics"] = []
            else:
                book["topics"] = []
            
            # Save to corpus
            save_book_to_corpus(book, DEFAULT_CORPUS_PATH)
            
            output = f"Successfully added book to corpus:\n\n"
            output += f"**{title}** by {author}\n"
            output += f"ID: {book_id}\n"
            if book.get("topics"):
                output += f"Extracted {len(book['topics'])} topics\n"
            
            return [TextContent(type="text", text=output)]
        
        elif name == "get_books_by_author":
            author_name = arguments["author_name"]
            
            results = search_books(author_name, books, search_type="author")
            
            if not results:
                return [TextContent(
                    type="text",
                    text=f"No books found by author '{author_name}'"
                )]
            
            output = f"Found {len(results)} book(s) by '{author_name}':\n\n"
            for book in results:
                output += f"- **{book.get('title', 'Untitled')}** (ID: {book.get('book_id')})\n"
                output += f"  Rating: {book.get('average_rating', 'N/A')}\n"
            
            return [TextContent(type="text", text=output)]
        
        elif name == "get_books_by_topic":
            topic_query = arguments["topic_query"].lower()
            min_confidence = arguments.get("min_confidence", 0.0)
            
            results = []
            for book in books:
                topics = book.get("topics", [])
                for topic in topics:
                    label = topic.get("label", "").lower()
                    category = topic.get("category", "").lower()
                    confidence = topic.get("confidence", 0.0)
                    
                    if (topic_query in label or topic_query in category) and confidence >= min_confidence:
                        results.append({
                            "book": book,
                            "topic": topic
                        })
                        break
            
            if not results:
                return [TextContent(
                    type="text",
                    text=f"No books found with topic '{topic_query}'"
                )]
            
            output = f"Found {len(results)} book(s) with topic '{topic_query}':\n\n"
            for item in results:
                book = item["book"]
                topic = item["topic"]
                output += f"- **{book.get('title', 'Untitled')}** by {book.get('author', 'Unknown')}\n"
                output += f"  Topic: {topic.get('label')} ({topic.get('category')}) - Confidence: {topic.get('confidence', 0):.2f}\n"
            
            return [TextContent(type="text", text=output)]
        
        elif name == "get_corpus_stats":
            total_books = len(books)
            
            # Count unique authors
            authors = set()
            for book in books:
                author = book.get("author")
                if author:
                    authors.add(author)
            
            # Count topics
            all_topics = set()
            all_categories = set()
            books_with_topics = 0
            
            for book in books:
                topics = book.get("topics", [])
                if topics:
                    books_with_topics += 1
                    for topic in topics:
                        all_topics.add(topic.get("label"))
                        all_categories.add(topic.get("category"))
            
            # Average rating
            ratings = [book.get("average_rating", 0) for book in books if book.get("average_rating")]
            avg_rating = sum(ratings) / len(ratings) if ratings else 0
            
            output = f"**Corpus Statistics:**\n\n"
            output += f"- Total books: {total_books}\n"
            output += f"- Unique authors: {len(authors)}\n"
            output += f"- Books with topics: {books_with_topics}\n"
            output += f"- Unique topics: {len(all_topics)}\n"
            output += f"- Unique categories: {len(all_categories)}\n"
            output += f"- Average rating: {avg_rating:.2f}\n"
            
            return [TextContent(type="text", text=output)]
        
        elif name == "import_book_from_csv":
            book_id = arguments["book_id"]
            extract_topics = arguments.get("extract_topics", True)
            
            # Check if already in corpus
            for b in books:
                if b.get("book_id") == book_id:
                    return [TextContent(
                        type="text",
                        text=f"Book with ID '{book_id}' already exists in corpus."
                    )]
            
            # Load CSV
            csv_books = load_original_csv()
            if not csv_books:
                return [TextContent(
                    type="text",
                    text="Error: Could not load original CSV file."
                )]
            
            # Find book in CSV
            csv_row = find_book_in_csv(book_id, csv_books)
            if not csv_row:
                return [TextContent(
                    type="text",
                    text=f"Book with ID '{book_id}' not found in original CSV."
                )]
            
            # Convert to corpus format
            book = csv_to_corpus_format(csv_row)
            
            # Extract topics if requested
            if extract_topics and book.get("description"):
                try:
                    title = book.get("title", "")
                    description = book.get("description", "")
                    
                    messages = _build_messages(title, description)
                    raw = run_llm(messages=messages, client=client, model="gpt-oss-120b")
                    topics_obj = _sanitize_and_validate_topics(raw)
                    
                    book["topics"] = topics_obj["topics"]
                    book["topics_ts"] = datetime.now(timezone.utc).isoformat()
                except Exception as e:
                    logger.error(f"Error extracting topics: {e}")
                    book["topics"] = []
            else:
                book["topics"] = []
            
            # Save to corpus
            save_book_to_corpus(book, DEFAULT_CORPUS_PATH)
            
            output = f"✅ Successfully imported book from CSV:\n\n"
            output += f"**{book.get('title')}** by {book.get('author')}\n"
            output += f"ID: {book_id}\n"
            output += f"Rating: {book.get('average_rating', 'N/A')}\n"
            if book.get("topics"):
                output += f"Extracted {len(book['topics'])} topics:\n"
                for topic in book["topics"][:3]:
                    output += f"  - {topic.get('label')} ({topic.get('category')})\n"
            
            return [TextContent(type="text", text=output)]
        
        else:
            return [TextContent(
                type="text",
                text=f"Unknown tool: {name}"
            )]
    
    except Exception as e:
        logger.error(f"Error in tool {name}: {e}", exc_info=True)
        return [TextContent(
            type="text",
            text=f"Error executing {name}: {str(e)}"
        )]


async def main():
    """Run the MCP server."""
    logger.info("Starting Goodreads Knowledge Graph MCP Server")
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
