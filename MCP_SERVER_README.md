# Goodreads Knowledge Graph MCP Server

An MCP server that allows Claude Desktop to interact with your Goodreads book knowledge graph.

## Features

The MCP server provides Claude with the following capabilities:

### 🔍 Search & Discovery
- **search_books**: Search for books by title or author
- **get_book_details**: Get comprehensive information about a specific book
- **get_books_by_author**: Find all books by an author
- **get_books_by_topic**: Discover books by topic or genre

### 🤖 AI-Powered Analysis
- **extract_book_topics**: Use LLM to extract hierarchical topics from book descriptions
- **generate_ego_graph**: Create interactive knowledge graph visualizations

### 📚 Corpus Management
- **add_book_to_corpus**: Add new books with automatic topic extraction
- **get_corpus_stats**: View statistics about your book collection

## Installation

1. **Install dependencies**:
```bash
cd /storage/homefs/as23z124/NLP_CAS_M3_project
pip install mcp
```

2. **Configure Claude Desktop**:

Add this to your Claude Desktop config file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "goodreads-knowledge-graph": {
      "command": "python3",
      "args": [
        "/storage/homefs/as23z124/NLP_CAS_M3_project/mcp_server_books.py"
      ],
      "env": {
        "PYTHONPATH": "/storage/homefs/as23z124/NLP_CAS_M3_project"
      }
    }
  }
}
```

3. **Restart Claude Desktop**

## Usage Examples

Once configured, you can ask Claude things like:

### Search for books
```
"Find books by J.K. Rowling"
"Search for books about artificial intelligence"
```

### Get book details
```
"Tell me more about the book with ID 6050894"
"What are the topics for Harry Potter?"
```

### Analyze books
```
"Extract topics from this book description: [paste description]"
"Show me the knowledge graph for book ID 6050894"
```

### Add new books
```
"Add a new book to the corpus:
Title: The Pragmatic Programmer
Author: Andrew Hunt
Description: [description]"
```

### Explore connections
```
"What books do we have about science fiction?"
"Show me all books in the Fantasy category"
"What are the corpus statistics?"
```

## Available Tools

| Tool | Description |
|------|-------------|
| `search_books` | Search by title or author |
| `get_book_details` | Get full book information |
| `extract_book_topics` | LLM-powered topic extraction |
| `generate_ego_graph` | Create knowledge graph visualization |
| `add_book_to_corpus` | Add new book with metadata |
| `get_books_by_author` | Find all books by an author |
| `get_books_by_topic` | Find books by topic/genre |
| `get_corpus_stats` | View corpus statistics |

## How It Works

1. **Claude Desktop** connects to the MCP server via stdio
2. Claude can **call tools** to interact with your book corpus
3. The server **loads data** from `goodreads_corpus/corpus_table.with_topics.jsonl`
4. Tools can:
   - Query existing books
   - Extract topics using your LLM
   - Generate knowledge graphs
   - Add new books to the corpus

## Architecture

```
Claude Desktop
    ↓ (MCP Protocol)
MCP Server (mcp_server_books.py)
    ↓
├── utils_KG.py (graph generation, topic extraction)
├── OpenAI Client (LLM for topics)
└── Corpus Files (JSONL book data)
```

## Troubleshooting

### Server not showing in Claude Desktop
- Check the config file path
- Verify Python path is correct
- Restart Claude Desktop
- Check logs in Claude Desktop dev tools

### Import errors
- Ensure `utils_KG.py` is in the same directory
- Check PYTHONPATH in config
- Install missing packages: `pip install networkx pyvis python-dotenv openai`

### .env file not found
- Ensure `.env` file exists in `/storage/homefs/as23z124/NLP_CAS_M3_project/`
- Contains: `BASE_URL` and `GPUSTACK_API_KEY`

## Configuration

Edit `mcp_server_books.py` to customize:

```python
# Default corpus path
DEFAULT_CORPUS_PATH = Path("/storage/homefs/as23z124/NLP_CAS_M3_project/goodreads_corpus")

# LLM model
model="gpt-oss-120b"
```

## Next Steps

Once working, you can:
1. Ask Claude to analyze books and extract insights
2. Have Claude build custom knowledge graphs
3. Let Claude discover connections between books
4. Use Claude to maintain and expand your corpus
5. Query across topics, authors, and recommendations

Enjoy exploring your book knowledge graph with Claude! 📚✨
