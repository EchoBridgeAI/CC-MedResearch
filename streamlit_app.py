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

# New: import the new modules for each pipeline phase
from query_generation import generate_query
from relevance_check import check_relevance
from summarization import summarize_article
from synthesis import synthesize_summaries
from citation_builder import build_references

# Configure environment and SSL
load_dotenv()

# Replace the config variables with:
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
NCBI_API_KEY = os.getenv('NCBI_API_KEY')
EMAIL = os.getenv('EMAIL')

# All configurations moved to streamlit_app.py
SEARCH_PARAMS = {
    'max_articles': 5,
    'min_articles': 1,
    'max_queries': 3,
    'max_retries': 2,
    'relevance_threshold': 0.7,
    'date_range_years': 10,
    'batch_size': 5
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
        'model': "chatgpt-4o-latest",
        'temperature': 0.3,
        'max_tokens': 100,
        'retry_delay': 2,
        'system_role': "You are a helpful expert medical researcher librarian.",
        'prompt_template': (
            "Based on this abstract, determine if the full text of this article likely contains information relevant to the clinical question.\n\n"
            "Question: {question}\n\n"
            "Abstract: {article_text}\n\n"
            "Consider:\n"
            "1. Direct relevance to the question\n"
            "2. Study type and methodology\n"
            "3. Population characteristics\n"
            "4. Intervention or exposure\n"
            "5. Outcomes measured\n\n"
            "Respond with only 'yes' or 'no'."
        )
    },
    'summarization': {
        'model': "chatgpt-4o-latest",
        'temperature': 0.5,
        'max_tokens': 500,
        'retry_delay': 5,
        'system_role': "You are a medical research assistant specializing in summarizing scientific articles.",
        'prompt_template': (
            "Summarize this article abstract in relation to the clinical question:\n\n"
            "Question: {question}\n\n"
            "Abstract: {article_text}\n\n"
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
        'model': "chatgpt-4o-latest",
        'temperature': 0.7,
        'max_tokens': 4096,
        'retry_delay': 10,
        'system_role': "You are a medical research assistant specializing in synthesizing findings from multiple studies.",
        'prompt_template': (
            "Synthesize these article summaries to answer the clinical question:\n\n"
            "Question: {question}\n\n"
            "Article Summaries:\n"
            "{article_summaries}\n\n"
            "Provide a comprehensive synthesis that:\n"
            "1. Begins with a brief overview\n"
            "2. Presents main findings, citing articles using their PMID numbers in square brackets [PMID:XXXXXX]\n"
            "3. Addresses contradictions or conflicts\n"
            "4. Notes limitations and gaps\n"
            "5. Concludes with clinical implications\n\n"
            "Important: \n"
            "- Use PMID numbers in square brackets [PMID:XXXXXX] to cite findings\n"
            "- Each article summary begins with \"Article PMID:XXXXXX\", use these exact PMID numbers in your citations\n"
            "- Do not include a references section, citations will be added automatically"
        )
    }
}

# Configure logging
class HTTPFilter(logging.Filter):
    def filter(self, record):
        return 'HTTP' not in record.getMessage()

# Create handlers
file_handler = logging.FileHandler('app.log')
console_handler = logging.StreamHandler()
console_handler.addFilter(HTTPFilter())

logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG level to see more details
    format='%(levelname)s - %(name)s - %(message)s',
    handlers=[
        file_handler,  # Detailed logs to file
        console_handler  # Filtered logs to terminal
    ]
)

# Reduce noise from other libraries
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Fix SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

# Set OpenAI API key in environment
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

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
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * SEARCH_PARAMS['date_range_years'])
        
        # Format dates for PubMed
        date_range = f"{start_date.strftime('%Y/%m/%d')}:{end_date.strftime('%Y/%m/%d')}[Date - Publication]"
        
        # Combine query with date range
        full_query = f"({query}) AND {date_range}"
        st.session_state['full_pubmed_query'] = full_query  # Store the full query
        
        handle = Entrez.esearch(db="pubmed", term=full_query, retmax=max_articles)
        record = Entrez.read(handle)
        id_list = record.get("IdList", [])
        
        if not id_list:
            return []
        
        ids = ",".join(id_list)
        handle = Entrez.efetch(db="pubmed", id=ids, rettype="xml")
        records = Entrez.read(handle)
        articles = records.get("PubmedArticle", [])
        
        logger.info(f"Retrieved {len(articles)} articles within date range {SEARCH_PARAMS['date_range_years']} years")
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

# Configure logging to file only for worker threads
worker_logger = logging.getLogger('worker')
worker_logger.setLevel(logging.INFO)
worker_handler = logging.FileHandler('worker.log')
worker_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
worker_logger.addHandler(worker_handler)

# Configure OpenAI client logging to file only
openai_logger = logging.getLogger('openai')
openai_logger.setLevel(logging.INFO)
# Remove all existing handlers
for handler in openai_logger.handlers[:]:
    openai_logger.removeHandler(handler)
openai_handler = logging.FileHandler('openai.log')
openai_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
openai_logger.addHandler(openai_handler)
# Prevent OpenAI logs from propagating to root logger
openai_logger.propagate = False

# Configure worker logger to not propagate to root
worker_logger.propagate = False

def process_article(article, question, llm_configs):
    """Process a single article"""
    try:
        # Extract the article abstract
        abstract_obj = article.get('MedlineCitation', {}).get('Article', {}).get('Abstract', {}).get('AbstractText', [])
        if isinstance(abstract_obj, list):
            abstract_text = " ".join(abstract_obj)
        else:
            abstract_text = abstract_obj
            
        if not abstract_text.strip():
            pmid = article.get('MedlineCitation', {}).get('PMID', 'Unknown')
            worker_logger.error(f"No abstract for PMID: {pmid}")
            return None, f"Article with PMID: {pmid} (no abstract)"
        
        # Perform relevance check
        is_relevant = check_relevance(question, abstract_text, llm_configs["relevance_check"])
        
        if not is_relevant:
            pmid = article.get('MedlineCitation', {}).get('PMID', 'Unknown')
            return None, f"Article with PMID: {pmid} (not relevant)"
        
        # Summarize the article
        summary = summarize_article(question, abstract_text, llm_configs["summarization"])
        pmid = article.get('MedlineCitation', {}).get('PMID', 'Unknown')
        summary_with_pmid = f"Article PMID:{pmid}\n{summary}"
        
        return summary_with_pmid, None
        
    except Exception as e:
        pmid = article.get('MedlineCitation', {}).get('PMID', 'Unknown')
        worker_logger.error(f"Error processing article PMID {pmid}: {str(e)}")
        return None, f"Article with PMID: {pmid} (processing error)"

def process_articles(articles, question, llm_configs, num_workers=8):
    """Process all articles in parallel"""
    summaries = []
    irrelevant = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_article, article, question, llm_configs)
            for article in articles
        ]
        
        for future in as_completed(futures):
            try:
                summary, error = future.result()
                if summary:
                    summaries.append(summary)
                if error:
                    irrelevant.append(error)
            except Exception as e:
                logger.error(f"Error processing article batch: {str(e)}")
                time.sleep(20)  # Wait if we hit rate limits
      
    return summaries, irrelevant

def main():
    st.title("🏥 CC Medical Research Assistant")
    st.write("Ask a medical research question and get answers based on PubMed articles.")

    # Configure logging to show in Streamlit
    log_container = st.empty()
    
    class StreamlitHandler(logging.Handler):
        def emit(self, record):
            log_entry = self.format(record)
            with log_container:
                st.text(log_entry)
    
    # Add Streamlit handler to logger
    streamlit_handler = StreamlitHandler()
    streamlit_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(streamlit_handler)

    # User input
    question = st.text_area("Enter your medical research question:", height=100)
    
    if st.button("Search and Analyze", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
            return
        
        start_time = time.time()

        try:
            with st.spinner("📝 Generating PubMed query..."):
                generated_query = generate_query(question, LLM_CONFIGS["query_generation"])
                st.info(f"Generated Query: \n {generated_query}")
            
            with st.spinner("🔍 Searching PubMed..."):
                search_results = search_pubmed(generated_query, SEARCH_PARAMS['max_articles'])
                if not search_results:
                    st.warning("No articles found for this query.")
                    return

            article_summaries = []
            irrelevant_articles = []
            # Process each retrieved article
            with st.spinner("📚 Processing articles..."):
                article_summaries, irrelevant_articles = process_articles(
                    search_results, question, LLM_CONFIGS
                )

            if not article_summaries:
                st.warning("No relevant articles were found or all article processing failed.")
                if irrelevant_articles:
                    st.write("Irrelevant or failed articles:")
                    for art in irrelevant_articles:
                        st.write(f"- {art}")
                return

            with st.spinner("✍️ Synthesizing findings..."):
                synthesis = synthesize_summaries(question, article_summaries, LLM_CONFIGS["synthesis"])
            
            # Build references from the synthesis text based on cited PMIDs
            references = build_references(synthesis, articles=search_results)
            
            end_time = time.time()
            st.info(f"⏱️ Total Processing Time: {end_time - start_time:.2f} seconds")

            # Display results in tabs
            tab1, tab2, tab3 = st.tabs(["Synthesis", "Article Summaries", "Search Details"])
            
            with tab1:
                st.markdown("### Synthesis")
                display_synthesis_and_references(synthesis, search_results)
            
            with tab2:
                st.markdown("### Article Summaries")
                for i, summary in enumerate(article_summaries, 1):
                    with st.expander(f"Article Summary {i}"):
                        st.markdown(summary)
            
            with tab3:
                st.markdown("### Search Details")
                st.markdown("**Base Query:**")
                st.markdown(f"```\n{generated_query}\n```")
                st.markdown("**Full PubMed Query (including date range):**")
                st.markdown(f"```\n{st.session_state.get('full_pubmed_query', 'Query not available')}\n```")
                if irrelevant_articles:
                    st.markdown("**Ignored Articles:**")
                    for art in irrelevant_articles:
                        st.markdown(f"- {art}")
                else:
                    st.markdown("No irrelevant articles detected.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
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

if __name__ == "__main__":
    # Run the async main function
    main() 