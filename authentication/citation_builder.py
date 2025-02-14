import logging

def format_reference(article):
    logger = logging.getLogger(__name__)
    
    try:
        # Get MedlineCitation safely
        citation = article.get('MedlineCitation')
        logger.info(f"Processing article data type: {type(article)}")
        logger.info(f"Available article keys: {list(article.keys()) if article else 'None'}")
        if citation:
            article_data = citation.get('Article', {})
            pmid = str(citation.get('PMID', ''))
            logger.info(f"Found PMID: {pmid}")
            logger.info(f"Article data keys: {list(article_data.keys()) if article_data else 'None'}")
            title = article_data.get('ArticleTitle', 'No title')
            
            # Build authors list if available
            authors = []
            author_list = article_data.get('AuthorList', [])
            author_count = len(author_list) if isinstance(author_list, list) else 0
            logger.info(f"Processing {author_count} authors")
            if isinstance(author_list, list):
                for author in author_list:
                    if isinstance(author, dict):
                        last_name = author.get('LastName')
                        initials = author.get('Initials')
                        if last_name and initials:
                            authors.append(f"{last_name} {initials}")
            authors_str = ', '.join(authors) if authors else "No author listed"
            if authors and len(authors) > 6:
                authors_str = ', '.join(authors[:6]) + ", et al."
            
            # Build publication details using Journal and Pagination data
            journal = article_data.get('Journal', {})
            journal_title = journal.get('Title', 'No journal')
            logger.info(f"Journal info: {journal_title}")
            journal_issue = journal.get('JournalIssue', {})
            pub_date = journal_issue.get('PubDate', {})
            year = pub_date.get('Year', '')
            volume = journal_issue.get('Volume', '')
            issue = journal_issue.get('Issue', '')
            pagination = article_data.get('Pagination', {}).get('MedlinePgn', '')
            
            # Attempt to extract DOI from ELocationID
            doi = ""
            eloc = article_data.get('ELocationID')
            if eloc:
                if isinstance(eloc, list):
                    for item in eloc:
                        if isinstance(item, dict) and item.get('EIdType') == 'doi':
                            doi = item.get('content', '')
                            break
                elif isinstance(eloc, dict):
                    if eloc.get('EIdType') == 'doi':
                        doi = eloc.get('content', '')
                else:
                    doi = str(eloc)
            
            pub_details = f"{journal_title}. {year}" if year else journal_title
            if volume:
                pub_details += f";{volume}"
            if issue:
                pub_details += f"({issue})"
            if pagination:
                pub_details += f":{pagination}"
            if doi:
                pub_details += f". DOI: {doi}"
            
            # Log the final reference data before returning
            logger.info(f"Formatted reference - PMID: {pmid}, Title: {title[:50]}...")
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "#"
            
            return {
                "url": url,
                "title": title,
                "authors": authors_str,
                "publication_details": pub_details
            }
        else:
            logger.error("Failed to find MedlineCitation. Article may be malformed.")
            logger.info(f"Attempting fallback with available data: {list(article.keys()) if article else 'None'}")
            # Fallback if 'MedlineCitation' is missing
            title = article.get('title', 'No title')
            pmid = article.get('pmid', '')
            logger.debug(f"Fallback PMID: {pmid}")
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "#"
            authors_str = article.get('authors', 'No author listed')
            pub_details = article.get('publication_details', 'No details')
            logger.info(f"Using fallback data - PMID: {pmid}, Title: {title[:50]}...")
            return {
                "url": url,
                "title": title,
                "authors": authors_str,
                "publication_details": pub_details
            }
    except Exception as e:
        logger.exception(f"Reference formatting error: {str(e)}")
        logger.error(f"Article data that caused error: {type(article)}")
        return {
            "url": "#",
            "title": "Error formatting reference",
            "authors": "",
            "publication_details": ""
        } 