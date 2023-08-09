from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
FAISS_PATH = 'vector/similarity_db'
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cuda'})
def store_vector(file_path):
    '''
    function will first load the CSV and converts into the embeddings
    by hugging face. FAISS is use to calculate the similarity of embeddings
    which will be store in a vector file
    '''
    loader = CSVLoader(file_path=file_path,
                      encoding="utf-8", csv_args={
                      'delimiter': '\t'})
    data = loader.load()
    db = FAISS.from_documents(data, embeddings)
    db.save_local(FAISS_PATH)

def load_vector():
    db = FAISS.load_local(FAISS_PATH, embeddings)
    return db

def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "drive/MyDrive/llama-2-7b-chat.ggmlv3.q4_0.bin",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0
    )
    return llm
