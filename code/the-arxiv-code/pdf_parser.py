from os import listdir
from os.path import isfile, join
import PyPDF2, arxiv
import pandas as pd
import refextract as ref

def main():
	path = "/home/chris/Documents/arXiv/pdf/"
	files = [f for f in listdir(path) if isfile(join(path, f))]
	
	q_results = []
	pagenum_lst = []
	refs = []
	
	for i in files:
		temp_file = open(path + i, 'rb')
		reader = PyPDF2.PdfFileReader(temp_file)
		front_page = reader.getPage(0).extractText()
		try:
			head = front_page.index(":") + 1
			tail = front_page.index("v", head)
			arxiv_id = front_page[head:tail]
			single_search = arxiv.query(search_query = arxiv_id)
			
			q_results.append(single_search[0])
			pagenum_lst.append(reader.numPages)
			refs.append(len(ref.extract_references_from_file(path + i)))
		except:
			pass
		temp_file.close()
			
	q_df = pd.DataFrame.from_dict(q_results)
	categories = ['authors', 'published_parsed', 'summary', 'arxiv_primary_category']
	df = q_df[categories].copy()
	
	weights = map(lambda y: 1.0 / (y - 1)
	                        if y != 1
	                        else 0,
	              map(lambda x: len(x), df['authors']))        
	categories = map(lambda x: x.get('term'), df['arxiv_primary_category'])
	        
	df['weights'] = pd.Series(weights)
	df['pagenum'] = pd.Series(pagenum_lst)
	df['refnum'] = pd.Series(refs)
	df['category'] = pd.Series(categories)
	
	df.to_csv('data.csv', index = False)
	
main()
