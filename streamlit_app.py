# streamlit run streamlit_app.py

import streamlit as st
import os
from dotenv import load_dotenv
import ssl
import logging
import time
import re
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from streamlit.runtime.scriptrunner import add_script_run_ctx
from logging.handlers import RotatingFileHandler
from logger_config import setup_logging

# New: import the new modules for each pipeline phase
from query_generation import generate_query
from relevance_check import check_relevance, extract_pmid_from_text
from summarization import summarize_article, fetch_full_text
from synthesis import synthesize_summaries
from citation_builder import build_references

# Configure environment and SSL
load_dotenv()

# Replace the config variables with:
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
NCBI_API_KEY = os.getenv('NCBI_API_KEY')
EMAIL = os.getenv('EMAIL')

# All configurations moved to streamlit_app.py
SEARCH_PARAMS = {
    'max_articles': 50,
    'min_articles': 1,
    'max_queries': 3,
    'max_retries': 2,
    'date_range_years': 10,
}

API_PARAMS = {
    'timeout': 30,
    'retry_delay': 5,
    'concurrent_calls': 3,
    'temperature': 0.7
}

TEXT_PARAMS = {
    'max_abstract_tokens': 1000,
    'max_summary_tokens': 500,
    'min_summary_tokens': 100,
    'truncate_threshold': 0.8
}

LLM_CONFIGS = {
    'query_generation': {
        'model': "chatgpt-4o-latest",
        'temperature': 0.7,
        'max_tokens': 1024,
        'retry_delay': 5,
        'system_role': "You are an expert medical librarian specializing in PubMed query construction.",
        'prompt_template': (
            "As a medical librarian, construct a precise PubMed search query for this clinical question:\n\n"
            "Question: {question}\n\n"
            "Guidelines:\n"
            "1. Use MeSH terms when possible\n"
            "2. Include relevant synonyms and abbreviations\n"
            "3. Use appropriate field tags [Title/Abstract], [MeSH Terms], etc.\n"
            "4. Structure with Boolean operators (AND, OR)\n"
            "5. Focus on key medical concepts\n"
            "6. Ensure comprehensive but specific coverage\n\n"
            "Return only the query, enclosed in triple backticks."
        )
    },
    'relevance_check': {
        'model': "llama-3.3-70b-versatile",
        'max_tokens': 100,
        'temperature': 0.1,
        'retry_delay': 2,
        'system_role': "You are a medical research assistant. Respond with only 'yes' or 'no'.",
        'prompt_template': (
            "Is this article abstract relevant to answering this question?\n\n"
            "Question: {question}\n\n"
            "Abstract: {article_text}\n\n"
            "Respond with only 'yes' or 'no'."
        )
    },
    'summarization': {
        'model': "llama-3.3-70b-versatile",
        'temperature': 0.5,
        'max_tokens': 500,
        'retry_delay': 5,
        'system_role': "You are a medical research assistant specializing in summarizing scientific articles.",
        'prompt_template': (
            "Summarize this article in relation to the clinical question:\n\n"
            "Question: {question}\n\n"
            "Article Text: {article_text}\n\n"
            "Provide a focused summary that:\n"
            "1. Highlights main findings and conclusions\n"
            "2. Includes methodology overview\n"
            "3. Notes population characteristics\n"
            "4. States key statistical results\n"
            "5. Explains clinical implications\n\n"
            "Be concise but include specific numbers and data."
        )
    },
    'synthesis': {
        'model': "o3-mini",
        'max_tokens': 8192,
        'retry_delay': 10,
        'reasoning_effort': "high",
        'system_role': (
            "You are a medical research assistant specializing in synthesizing findings from multiple studies. "
            "Format your response using markdown for better readability:\n"
            "- Use ## for section headings\n"
            "- Use **bold** for emphasis of key findings\n"
            "- Use bullet points for lists\n"
            "- Use > for important quotes or key statistics"
        ),
        'prompt_template': (
            "Synthesize these article summaries to answer the clinical question using markdown formatting:\n\n"
            "Question: {question}\n\n"
            "Article Summaries:\n"
            "{article_summaries}\n\n"
            "Structure your response with these sections:\n\n"
            "## Overview\n"
            "Provide a brief overview of the current evidence.\n\n"
            "## Main Findings\n"
            "Present key findings, using bullet points where appropriate. "
            "Cite articles using PMID numbers [PMID:XXXXXX]. "
            "Use **bold** for emphasis of important findings.\n\n"
            "## Contradictions and Conflicts\n"
            "Discuss any contradictory findings or conflicting evidence between studies.\n\n"
            "## Limitations\n"
            "Address study limitations and gaps in current evidence.\n\n"
            "## Clinical Implications\n"
            "Conclude with practical implications for clinical practice.\n\n"
            "Important Guidelines:\n"
            "- Use markdown formatting consistently\n"
            "- Always cite findings with PMID numbers in square brackets [PMID:XXXXXX]\n"
            "- Use bullet points for listing multiple findings\n"
            "- Use > for highlighting key statistics or quotes\n"
            "- Use **bold** for emphasis of critical points\n"
            "- Each article summary begins with \"Article PMID:XXXXXX\", use these exact PMID numbers\n"
            "- Do not include a references section, citations will be added automatically"
        )
    }
}

# Initialize logger at the start of the app
logger = setup_logging()

# Fix SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

# Set OpenAI API key in environment
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# Set Groq API key in environment
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Page config
st.set_page_config(
    page_title="CC Medical Research Assistant",
    page_icon="🏥",
    layout="wide"
)

# New: A helper function to search PubMed using Biopython's Entrez.
def search_pubmed(query, max_articles):
    try:
        from Bio import Entrez
    except ImportError:
        st.error("Please install Biopython for PubMed search functionality.")
        return []
    
    Entrez.email = EMAIL
    if NCBI_API_KEY:
        Entrez.api_key = NCBI_API_KEY
    
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_articles)
        record = Entrez.read(handle)
        id_list = record.get("IdList", [])
        
        if not id_list:
            return []
        
        ids = ",".join(id_list)
        handle = Entrez.efetch(db="pubmed", id=ids, rettype="xml")
        records = Entrez.read(handle)
        articles = records.get("PubmedArticle", [])
        
        logger.info(f"Retrieved {len(articles)} articles")
        return articles
        
    except Exception as e:
        logger.error(f"Error searching PubMed: {str(e)}")
        return []

# CSS for citations and references
CITATION_CSS = """
<style>
    .citation-link {
        color: #0366d6;
        text-decoration: none;
    }
    .citation-link:hover {
        text-decoration: underline;
    }
    .reference-entry {
        margin-bottom: 1em;
        padding: 0.5em;
        border-left: 3px solid #e1e4e8;
    }
    .reference-entry:hover {
        background-color: #f6f8fa;
    }
</style>
"""

def display_synthesis_and_references(synthesis_text, articles):
    """Display the synthesis with clickable citations and formatted references"""
    try:
        # Add CSS
        st.markdown(CITATION_CSS, unsafe_allow_html=True)
        
        # Process synthesis and get references
        processed_text, references = build_references(synthesis_text, articles)
        
        if not references:
            st.warning("No references found in the synthesis.")
            st.markdown(synthesis_text)
            return
        
        # Display synthesis with clickable citations
        st.markdown(processed_text, unsafe_allow_html=True)
        
        # Display references section
        if references:
            st.markdown("---")
            st.markdown("### References")
            # Combine all reference HTML
            references_html = "\n".join(references)
            st.markdown(references_html, unsafe_allow_html=True)
            
    except Exception as e:
        logger.exception("Error displaying synthesis and references")
        st.error("Error displaying synthesis and references. Please try again.")
        st.markdown(synthesis_text)

def check_relevance(question, article_text, config):
    """Check if an article is relevant to the research question"""
    try:
        from groq import Groq
        client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        
        # Use logger instead of st.text
        logger.info("Making relevance check API call to Groq")
        
        messages = [
            {
                "role": "system",
                "content": "You are a medical research assistant. Respond with only 'yes' or 'no'."
            },
            {
                "role": "user",
                "content": f"Is this article abstract relevant to answering this question?\n\nQuestion: {question}\n\nAbstract: {article_text}\n\nRespond with only 'yes' or 'no'."
            }
        ]
        
        try:
            response = client.chat.completions.create(
                messages=messages,
                model=config['model'],
                max_tokens=config['max_tokens'],
                temperature=0.1
            )
            
            # Log response
            logger.info(f"Received response from Groq: {response.choices[0].message.content}")
            
            answer = response.choices[0].message.content.strip().lower()
            
            # More robust response handling
            if answer and 'yes' in answer:
                return True
            elif answer and 'no' in answer:
                return False
            else:
                logger.warning(f"Unexpected response from Groq: {answer}")
                return True  # Default to including article if response is unclear
                
        except Exception as api_error:
            logger.error(f"Groq API error: {str(api_error)}")
            return True  # Default to including article on API error
            
    except Exception as e:
        logger.error(f"Error in relevance check: {str(e)}", exc_info=True)
        return True

def process_article(article, question, llm_configs):
    """Process a single article"""
    pmid = article.get('MedlineCitation', {}).get('PMID', 'Unknown')
    logger.info(f"Processing article PMID: {pmid}")
    
    try:
        # Extract the article abstract
        abstract_obj = article.get('MedlineCitation', {}).get('Article', {}).get('Abstract', {}).get('AbstractText', [])
        
        if isinstance(abstract_obj, list):
            abstract_text = " ".join(str(text) for text in abstract_obj)
        else:
            abstract_text = str(abstract_obj)
            
        logger.info(f"PMID {pmid}: Abstract length: {len(abstract_text)}")
            
        if not abstract_text or not abstract_text.strip():
            logger.error(f"No abstract content for PMID: {pmid}")
            return None, f"Article PMID:{pmid} (no abstract)"
        
        # Add rate limiting
        time.sleep(0.5)  # Add a small delay between API calls
        
        # Add logging to debug relevance checking
        logger.info(f"Starting relevance check for PMID: {pmid}")
        try:
            is_relevant = check_relevance(question, abstract_text, llm_configs["relevance_check"])
            logger.info(f"Relevance check result for PMID {pmid}: {is_relevant}")
        except Exception as rel_error:
            logger.error(f"Relevance check failed for PMID {pmid}: {str(rel_error)}")
            is_relevant = True  # Default to including on error
        
        if not is_relevant:
            return None, f"Article PMID:{pmid} (not relevant)"
        
        # Process relevant articles
        logger.info(f"Fetching full text for PMID {pmid}")
        full_text = fetch_full_text(pmid)
        article_text = full_text if full_text else abstract_text
        
        # Add rate limiting for summarization
        time.sleep(0.5)
        
        logger.info(f"Starting summarization for PMID {pmid}")
        try:
            summary = summarize_article(question, article_text, llm_configs["summarization"])
            summary_with_pmid = f"Article PMID:{pmid}\n{summary}"
            logger.info(f"Successfully summarized PMID {pmid}")
            return summary_with_pmid, None
        except Exception as sum_error:
            logger.error(f"Summarization failed for PMID {pmid}: {str(sum_error)}")
            raise
        
    except Exception as e:
        logger.error(f"Error processing PMID {pmid}: {str(e)}", exc_info=True)
        return None, f"Article PMID:{pmid} (processing error: {str(e)})"

def process_articles(articles, question, llm_configs, num_workers=8):
    """Process all articles in parallel"""
    summaries = []
    irrelevant = []

    # Suppress the Streamlit thread context warning
    warnings.filterwarnings('ignore', message='.*missing ScriptRunContext.*')

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all articles for parallel processing
        futures = [
            executor.submit(process_article, article, question, llm_configs)
            for article in articles
        ]

        # Collect results as they complete
        for future in as_completed(futures):
            try:
                summary, error = future.result()
                if summary:
                    summaries.append(summary)
                if error:
                    irrelevant.append(error)
            except Exception as e:
                logger.error("Error processing article batch:", exc_info=True)
                logger.error(f"Error details: {str(e)}")
                # Reduce the number of concurrent workers if we hit an error
                num_workers = max(1, num_workers - 1)
                time.sleep(5)  # Shorter wait time between retries

    if not summaries and irrelevant:
        logger.warning("No successful article processing. Errors encountered:")
        for err in irrelevant:
            logger.warning(err)

    return summaries, irrelevant

def display_logs():
    """Display logs in the Streamlit interface"""
    try:
        with open('logs/research_assistant.log', 'r') as f:
            logs = f.read()
        if logs:
            st.text_area("Application Logs", logs, height=400)
        else:
            st.info("No logs available for this session")
    except Exception as e:
        st.warning(f"Could not read log file: {str(e)}")

def get_date_range(years_back):
    """Generate PubMed date range filter for the last N years"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years_back)
    return f"{start_date.strftime('%Y/%m/%d')}:{end_date.strftime('%Y/%m/%d')}[Date - Publication]"

def create_progress_display():
    """Create and return containers for progress updates"""
    progress = {
        'query': st.empty(),      # Create query container first
        'search': st.empty(),     # Create search container second
        'processing': st.empty(),
        'synthesis': st.empty(),
        'references': st.empty(),
        'time': st.empty()
    }
    return progress

def update_progress(progress, step, message, status="running"):
    """Update progress display for a specific step"""
    emoji_map = {
        "running": "⏳",
        "success": "✅",
        "error": "❌",
        "warning": "⚠️",
        "excluded": "❌"
    }
    emoji = emoji_map.get(status, "")  # Default to no emoji instead of ℹ️
    progress[step].info(f"{emoji} {message}")

def extract_pmid_from_text(text):
    """Extract PMID from article text or summary"""
    match = re.search(r"PMID:(\d+)", text)
    return match.group(1) if match else "Unknown"

def main():
    # Get password from environment variables
    correct_password = os.getenv('APP_PASSWORD')
    if not correct_password:
        st.error("No password set. Please set APP_PASSWORD in .env file")
        return
    
    # Initialize session state for authentication
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        
    # Show password input only if not authenticated
    if not st.session_state.authenticated:
        password = st.text_input("Enter password", type="password")
        
        if not password:
            st.warning("Please enter the password to access the application")
            return
            
        if password != correct_password:
            st.error("Incorrect password")
            return
        
        st.session_state.authenticated = True
        st.rerun()
    
    # Rest of your main function code starts here
    st.title("🏥 CC Medical Research Assistant")
    st.write("Ask a medical research question and get answers based on PubMed articles.")
    
    # Add sidebar for configuration
    with st.sidebar:
        st.header("Search Parameters")
        
        # Number of articles slider - max 100, step by 5
        max_articles = st.slider(
            "Number of articles to retrieve",
            min_value=5,
            max_value=100,
            value=SEARCH_PARAMS['max_articles'],
            step=5,
            help="Maximum number of articles to retrieve from PubMed"
        )
        
        # Date range slider - back to 1966
        current_year = datetime.now().year
        years_back = current_year - 1966
        
        date_range = st.slider(
            "Publication date cutoff (years)",
            min_value=1,
            max_value=years_back,
            value=SEARCH_PARAMS['date_range_years'],
            help="Only include articles published within this many years"
        )
        
        # Show the actual year range
        start_year = current_year - date_range
        st.caption(f"Will search from {start_year} to {current_year}")
    
    # Update the search parameters with user values
    SEARCH_PARAMS.update({
        'max_articles': max_articles,
        'date_range_years': date_range
    })
    
    # Rest of the main function...
    question = st.text_area("Enter your clinical question:", height=100)
    
    if st.button("Search") and question:
        try:
            progress = create_progress_display()
            start_time = time.time()
            
            # Query Generation Step
            update_progress(progress, 'query', "Generating PubMed query...", "running")
            generated_query = generate_query(question, LLM_CONFIGS["query_generation"])
            date_range = get_date_range(SEARCH_PARAMS['date_range_years'])
            full_query = f"({generated_query}) AND {date_range}"
            update_progress(progress, 'query', f"⚙️ Generated query:\n\n```\n{full_query}\n```", "")
            
            # PubMed Search Step
            update_progress(progress, 'search', "Searching PubMed...", "running")
            search_results = search_pubmed(full_query, SEARCH_PARAMS['max_articles'])
            article_count = len(search_results) if search_results else 0
            
            if not search_results:
                update_progress(progress, 'search', "❌ No articles found", "error")
                return
            update_progress(progress, 'search', f"🔍 Found {article_count} articles", "")
            
            # Article Processing Step
            update_progress(progress, 'processing', "Processing articles...", "running")
            article_summaries, irrelevant_articles = process_articles(
                search_results, question, LLM_CONFIGS
            )
            relevant_count = len(article_summaries)
            irrelevant_count = len(irrelevant_articles)
            
            if not article_summaries:
                update_progress(progress, 'processing', "⚠️ No relevant articles found", "warning")
                return
            
            # Keep existing emojis for relevant/irrelevant counts
            processing_message = []
            if relevant_count > 0:
                processing_message.append(f"✅ {relevant_count} relevant articles found")
            if irrelevant_count > 0:
                processing_message.append(f"❌ {irrelevant_count} articles excluded")
            update_progress(progress, 'processing', "\n".join(processing_message), "")
            
            # Synthesis Step
            update_progress(progress, 'synthesis', "Synthesizing findings...", "running")
            synthesis = synthesize_summaries(question, article_summaries, LLM_CONFIGS["synthesis"])
            update_progress(progress, 'synthesis', "✍️ Synthesis complete", "")
            
            # References Step
            update_progress(progress, 'references', "Building references...", "running")
            references = build_references(synthesis, articles=search_results)
            update_progress(progress, 'references', "📝 References compiled", "")
            
            # Time tracking
            end_time = time.time()
            processing_time = end_time - start_time
            update_progress(progress, 'time', f"⏱️ Total processing time: {processing_time:.1f} seconds", "")
            
            # Display results in tabs
            tab1, tab2, tab3 = st.tabs(["Synthesis", "Article Summaries", "Search Details"])
            
            with tab1:
                display_synthesis_and_references(synthesis, search_results)
            
            with tab2:
                st.markdown("### Article Summaries")
                for i, summary in enumerate(article_summaries, 1):
                    # Extract PMID from summary
                    pmid = extract_pmid_from_text(summary)
                    
                    # Get article title from search_results
                    article = next(
                        (art for art in search_results 
                         if str(art.get('MedlineCitation', {}).get('PMID', '')) == pmid),
                        None
                    )
                    
                    if article:
                        title = article.get('MedlineCitation', {}).get('Article', {}).get('ArticleTitle', 'No title')
                        pubmed_link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                        
                        # Create expander with footnote number and title
                        with st.expander(f"[{i}] {title}"):
                            # Add PubMed link at the top of the summary
                            st.markdown(f"[View on PubMed]({pubmed_link})")
                            st.markdown("---")  # Add a separator
                            st.markdown(summary)
            
            with tab3:
                st.markdown("### Search Details")
                st.markdown("**PubMed Query:**")
                st.code(full_query, language="text")
                if irrelevant_articles:
                    st.markdown("**Excluded Articles:**")
                    for article_info in irrelevant_articles:
                        pmid = extract_pmid_from_text(article_info)
                        # Get article title from search_results
                        title = next(
                            (article.get('MedlineCitation', {}).get('Article', {}).get('ArticleTitle', 'No title')
                            for article in search_results
                            if article.get('MedlineCitation', {}).get('PMID', '') == pmid),
                            'No title'
                        )
                        pubmed_link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                        st.markdown(f"❌ [{title}]({pubmed_link})")

        except Exception as e:
            error_step = next((step for step in progress if not progress[step].empty()), 'processing')
            update_progress(progress, error_step, f"Error: {str(e)}", "error")
            logger.error("Error in main process", exc_info=True)

    # First, add custom CSS for the link color and gray text
    st.markdown("""
    <style>
    .reference-title { color: #E95420; text-decoration: none; }
    .reference-title:hover { text-decoration: underline; }
    .journal-info { color: #666666; }
    .citation-link { 
        color: #0066cc; 
        text-decoration: none;
        transition: color 0.2s ease;
    }
    .citation-link:hover { 
        color: #003366;
        text-decoration: underline;
    }
    .reference-item {
        scroll-margin-top: 2em;  /* Space for smooth scrolling */
        padding: 0.5em 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Add display logs button
    if st.button("Display Logs"):
        display_logs()

if __name__ == "__main__":
    # Run the async main function
    main() 