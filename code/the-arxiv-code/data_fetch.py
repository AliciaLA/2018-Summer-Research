import arxiv, urllib2, PyPDF2, refextract
import io, sys, os, time
import pandas as pd

#warning: not very reliable. working on a better version of the script

def main():
    subject = "topological data analysis"
    q_results = arxiv.query(search_query = subject, max_results = 1000)
	
    #initial query
    q_df = pd.DataFrame.from_dict(q_results)
	
    #query processing
    categories = ['authors', 'published_parsed', 'updated_parsed']
    p_df = q_df[categories].copy()
    
    #add more columns for more dimension
    p_df = custom_columns(q_df, p_df)
    
    #output to csv file
    p_df.to_csv('data.csv', index = False)
    
# add more columns to the dataframe
# post: p_df modified with more columns added
def custom_columns(q_df, p_df):
    p_df['primary_category'] = pd.Series(map(lambda x: x.get('term'),
                                        q_df['arxiv_primary_category']))

    p_df['tagged_categories'] = pd.Series(map(lambda x:
                                          map(lambda y:
                                              y.get('term'), x),
                                              q_df['tags']))
                                              
    p_df['weights'] = pd.Series(map(lambda y: 1.0 / (y - 1)
                                              if y != 1
                                              else 0,
                                    map(lambda x: len(x),
                                        p_df['authors'])))
	
    p_df['pages'] = pd.Series(map(lambda x: url_to_page(x),
                                  q_df['pdf_url'].copy()))
	                                      
    p_df['references'] = pd.Series(map(lambda x:
                         len(refextract.extract_references_from_url(x)),
                         q_df['pdf_url'].copy()))
    return p_df
	
# warning : slow
# converts pdf url to page numbers
def url_to_page(link):
	req = urllib2.Request(link)
	remote_file = urllib2.urlopen(req).read()
	memory_file = io.BytesIO(remote_file)
	try:
		temp_pdf = PyPDF2.PdfFileReader(memory_file)
	except PyPDF2.utils.PdfReadError:
		return 0
	return temp_pdf.getNumPages()

#processing time for reference
start_time = time.time()
main()
print("--- %s seconds ---" % (time.time() - start_time))
