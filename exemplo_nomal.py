from pathlib import Path

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

load_dotenv()


def importar_documentos(diretorio):
    docs_caminho = Path(diretorio)
    if not docs_caminho.exists():
        raise FileNotFoundError(f"Pasta '{diretorio}' não encontrado.")

    documentos = []
    for arquivo in docs_caminho.rglob('*.txt'):
        carregado = TextLoader(str(arquivo), encoding="utf-8")
        documentos.extend(carregado.load())
    return documentos


def criar_vector_store(_diretorio, _documentos_divididos):
    _embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    _vectorstore = FAISS.from_documents(_documentos_divididos, _embeddings)
    _vectorstore.save_local(_diretorio)


def ler_vector_store(_diretorio):
    _embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return FAISS.load_local(_diretorio, _embeddings, allow_dangerous_deserialization=True)


def dividir_documentos_chunks(_documentos):
    _splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    _documentos_divididos = _splitter.split_documents(_documentos)
    return _documentos_divididos


def preparar_documentos(_diretorio):
    docs_caminho = Path(_diretorio)
    if not docs_caminho.exists():
        _documentos = importar_documentos('docs')
        _documentos_divididos = dividir_documentos_chunks(_documentos)
        criar_vector_store(_diretorio, _documentos_divididos)


def exemplo_rag_raptor(_diretorio, _pergunta):
    preparar_documentos(_diretorio)
    vectorstore = ler_vector_store(_diretorio)

    llm = ChatOpenAI(temperature=0)
    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(),
        llm=llm
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

    result = qa_chain.invoke(_pergunta)
    print("\n Resposta:\n", result['result'])

    for doc in result["source_documents"]:
        print(doc.metadata.get("source", "desconhecido"))
        print(doc.page_content[:200], "...\n")

