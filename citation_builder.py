import re
import logging
from string import Template

logger = logging.getLogger(__name__)

# HTML templates for citations and references
CITATION_TEMPLATE = 'PMID:{pmid}<a href="#ref-{pmid}" class="citation-link">[{ref_num}]</a>'

REFERENCE_TEMPLATE = '''
<div id="ref-{pmid}" class="reference-entry">
    <p><strong>{ref_num}.</strong> <a href="{url}" target="_blank">{title}</a><br>
    {authors}<br>
    {publication_details}</p>
</div>
'''

__all__ = ['build_references', 'format_reference']

def build_references(synthesis_text, articles):
    """
    Process synthesis text and articles to generate formatted references.
    Returns (processed_text, reference_list)
    """
    try:
        # Initialize processed_text with input text
        processed_text = synthesis_text

        # Extract PMIDs from synthesis text, handling multiple PMIDs in brackets
        pmid_matches = re.finditer(r"\[((?:PMID:\d+(?:\s*,\s*)?)+)\]", synthesis_text)
        if not pmid_matches:
            logger.warning("No PMID citations found in synthesis text")
            return synthesis_text, []

        # Build mapping of PMIDs to reference numbers and articles
        pmid_map = {}
        ref_num = 1
        for match in pmid_matches:
            # Extract all PMIDs from the citation group
            pmids = re.findall(r"PMID:(\d+)", match.group(1))
            original_citation = match.group(0)  # The full matched text
            
            # Process each PMID in the group
            citation_refs = []
            for pmid in pmids:
                if pmid not in pmid_map:
                    # Find corresponding article
                    article = next((art for art in articles 
                                  if str(art.get('MedlineCitation', {}).get('PMID', '')) == pmid 
                                  or str(art.get('pmid', '')) == pmid), None)
                    if article:
                        pmid_map[pmid] = {
                            'ref_num': ref_num,
                            'article': article
                        }
                        ref_num += 1
                if pmid in pmid_map:
                    citation_refs.append(CITATION_TEMPLATE.format(
                        pmid=pmid,
                        ref_num=pmid_map[pmid]['ref_num']
                    ))
            
            # Replace the original citation with comma-separated references
            if citation_refs:
                processed_text = processed_text.replace(
                    original_citation, 
                    '[' + ', '.join(citation_refs) + ']'
                )

        if not pmid_map:
            logger.warning("No matching articles found for citations")
            return synthesis_text, []

        # Generate reference list
        references = []
        for pmid, data in sorted(pmid_map.items(), key=lambda x: x[1]['ref_num']):
            ref_data = format_reference(data['article'])
            if ref_data:
                ref_html = REFERENCE_TEMPLATE.format(
                    pmid=pmid,
                    ref_num=data['ref_num'],
                    **ref_data
                )
                references.append(ref_html)

        return processed_text, references

    except Exception as e:
        logger.exception("Error building references")
        return synthesis_text, []

def format_reference(article):
    """Format a single reference. Returns dict with reference data or None if error."""
    try:
        citation = article.get('MedlineCitation', {})
        if citation:
            article_data = citation.get('Article', {})
            pmid = str(citation.get('PMID', ''))
            
            # Get title and URL
            title = article_data.get('ArticleTitle', 'No title')
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "#"
            
            # Format authors
            authors = []
            for author in article_data.get('AuthorList', []):
                if isinstance(author, dict):
                    last_name = author.get('LastName', '')
                    initials = author.get('Initials', '')
                    if last_name and initials:
                        authors.append(f"{last_name} {initials}")
            
            authors_str = ', '.join(authors[:6])
            if len(authors) > 6:
                authors_str += ", et al."
            elif not authors:
                authors_str = "No author listed"

            # Format publication details
            journal = article_data.get('Journal', {})
            pub_details = []
            
            # Journal title and year
            journal_title = journal.get('Title', '')
            if journal_title:
                pub_details.append(journal_title)
            
            year = journal.get('JournalIssue', {}).get('PubDate', {}).get('Year', '')
            if year:
                pub_details.append(year)

            # Volume, issue, pages
            volume = journal.get('JournalIssue', {}).get('Volume', '')
            issue = journal.get('JournalIssue', {}).get('Issue', '')
            pages = article_data.get('Pagination', {}).get('MedlinePgn', '')
            
            if volume:
                pub_details.append(f"{volume}")
                if issue:
                    pub_details[-1] += f"({issue})"
            if pages:
                pub_details.append(f":{pages}")

            # Extract DOI from ELocationID
            eloc_list = article_data.get('ELocationID', [])
            if not isinstance(eloc_list, list):
                eloc_list = [eloc_list]
            
            doi = ''
            for eloc in eloc_list:
                if hasattr(eloc, 'attributes') and eloc.attributes.get('EIdType') == 'doi':
                    doi = str(eloc)
                    break
            
            # Add identifiers on new line
            identifiers = []
            if pmid:
                identifiers.append(f"PMID: {pmid}")
            if doi:
                identifiers.append(f"DOI: {doi}")
            
            # Join publication details and add identifiers on new line
            pub_details_str = '. '.join(pub_details)
            if identifiers:
                pub_details_str += f"\n{'. '.join(identifiers)}"

            return {
                'url': url,
                'title': title,
                'authors': authors_str,
                'publication_details': pub_details_str
            }

        else:
            # Fallback for non-MedlineCitation format
            return {
                'url': f"https://pubmed.ncbi.nlm.nih.gov/{article.get('pmid', '')}/",
                'title': article.get('title', 'No title'),
                'authors': article.get('authors', 'No author listed'),
                'publication_details': article.get('publication_details', '')
            }

    except Exception as e:
        logger.exception(f"Error formatting reference: {str(e)}")
        return None 