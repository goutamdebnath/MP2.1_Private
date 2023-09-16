import math
import sys
import time

import metapy
import pytoml

class InL2Ranker(metapy.index.RankingFunction):
    """
    Create a new ranking function in Python that can be used in MeTA.
    """
    def __init__(self, some_param=1.0):
        self.param = some_param
        # You *must* call the base class constructor here!
        super(InL2Ranker, self).__init__()

    def score_one(self, sd):
        """
        You need to override this function to return a score for a single term.
        For fields available in the score_data sd object,
        @see https://meta-toolkit.org/doxygen/structmeta_1_1index_1_1score__data.html
        """
        #return (self.param + sd.doc_term_count) / (self.param * sd.doc_unique_terms + sd.doc_size)
        tfn = sd.doc_term_count * math.log((1.0 + (sd.avg_dl / sd.doc_size)), 2)
        k = sd.query_term_weight*(tfn/(tfn+self.param)*math.log(((sd.num_docs+1)/sd.corpus_term_count+.5), 2))
        return k


def load_ranker(cfg_file):
    """
    Use this function to return the Ranker object to evaluate, e.g. return InL2Ranker(some_param=1.0) 
    The parameter to this function, cfg_file, is the path to a
    configuration file used to load the index. You can ignore this for MP2.
    """
    #return metapy.index.JelinekMercer()
    return InL2Ranker(some_param)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: {} config.toml".format(sys.argv[0]))
        sys.exit(1)

    cfg = sys.argv[1]
    print('Building or loading index...')
    idx = metapy.index.make_inverted_index(cfg)
    ranker = load_ranker(cfg)
    ev = metapy.index.IREval(cfg)

    with open(cfg, 'r') as fin:
        cfg_d = pytoml.load(fin)

    query_cfg = cfg_d['query-runner']
    if query_cfg is None:
        print("query-runner table needed in {}".format(cfg))
        sys.exit(1)

    start_time = time.time()
    top_k = 10
    query_path = query_cfg.get('query-path', 'queries.txt')
    query_start = query_cfg.get('query-id-start', 0)

    query = metapy.index.Document()
    print('Running queries')
    with open(query_path) as query_file:
        for query_num, line in enumerate(query_file):
            query.content(line.strip())
            results = ranker.score(idx, query, top_k)
            avg_p = ev.avg_p(results, query_start + query_num, top_k)
            print("Query {} average precision: {}".format(query_num + 1, avg_p))
    print("Mean average precision: {}".format(ev.map()))
    print("Elapsed: {} seconds".format(round(time.time() - start_time, 4)))

    #BM25
    query=metapy.index.Document()
    print ("****************** BM25 MAP ******************")
    for count in range(0,1):
        k1=1.2
        b=0.75
        k3=500
        ranker=metapy.index.OkapiBM25(k1,b,k3)
        ev = metapy.index.IREval(cfg)
        with open(cfg, 'r') as fin:
            cfg_d = pytoml.load(fin)
        query_cfg = cfg_d['query-runner']
        if query_cfg is None:
            print("query-runner table needed in {}".format(cfg))
            sys.exit(1)

        start_time = time.time()
        top_k = 10
        query_path = query_cfg.get('query-path', 'queries.txt')
        query_start = query_cfg.get('query-id-start', 0)
        num_results=10
        f=open("bm25.avg_p.txt", "w")
        with open('cranfield-queries.txt') as query_file:
            for query_num, line in enumerate(query_file):
                query.content(line.strip())
                results = ranker.score(idx, query, num_results)
                avg_p = ev.avg_p(results, query_start+query_num,num_results)
                f.write(str(avg_p)+str("\n"))
        
        print("Mean average precision: {} with k1 = {} b = {} k2 = {}".format(ev.map(), k1,b,k3))
        f.close()

        bm25=[]
        f1=open("bm25.avg_p.txt", "r")
        for line in f1:
            bm25.append(float(line.strip('\n')))
        inl2=[]
        f2=open("inl2.avg_p.txt", "r")
        for line in f2:
            inl2.append(float(line.strip('\n')))

        from scipy import stats
        tstat, pval = stats.ttest_rel(bm25, inl2)
        s1=open("significance.txt", "w")
        s1.write(str(pval))       
        print("pval = ", pval)