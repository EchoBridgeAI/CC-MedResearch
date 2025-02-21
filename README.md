# CC Medical Research Assistant

A Streamlit application that helps medical professionals search and synthesize research articles from PubMed using AI assistance.

## Features

- PubMed article search and retrieval
- AI-powered query generation
- Relevance checking of articles
- Article summarization
- Research synthesis
- Citation management

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_key_here
   GROQ_API_KEY=your_key_here
   NCBI_API_KEY=your_key_here
   EMAIL=your_email_here
   APP_PASSWORD=your_password_here
   ```
4. Run the application:
   ```bash
   streamlit run streamlit_app.py
   ```

## Usage

1. Enter your clinical question in the text area
2. Adjust search parameters in the sidebar if needed
3. Click "Search" to start the research process
4. View results in the Synthesis, Article Summaries, and Search Details tabs

## License

[Your chosen license] 