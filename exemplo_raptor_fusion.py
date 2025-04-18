from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.document_compressors import LLMChainExtractor, EmbeddingsFilter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from dotenv import load_dotenv
import os


load_dotenv()


# 1. Carrega todos os documentos .txt da pasta "documentos"
def carregar_documentos(caminho_pasta):
    documentos = []
    for nome_arquivo in os.listdir(caminho_pasta):
        if nome_arquivo.endswith(".txt"):
            loader = TextLoader(os.path.join(caminho_pasta, nome_arquivo), encoding="utf-8")
            documentos.extend(loader.load())
    return documentos

# 2. Carregar e dividir os documentos em chunks
documentos = carregar_documentos("docs")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(documentos)

# 3. Criar o vectorstore FAISS
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

# 4. Configurar o retriever com RAG Fusion (MultiQueryRetriever)
llm = OpenAI(temperature=0)
multiquery_retriever = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(), llm=llm)

# 5. Aplicar compressão com RAPTOR (Pipeline: embeddings filter + chain extractor)
compressor = DocumentCompressorPipeline(
    transformers=[
        EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76),
        LLMChainExtractor.from_llm(llm)
    ]
)

# 6. Combinar RAG Fusion + RAPTOR com ContextualCompressionRetriever
retriever_fusion_raptor = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=multiquery_retriever
)

# 7. Construir a chain final para responder perguntas
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever_fusion_raptor,
    return_source_documents=True
)

# 8. Fazer uma pergunta
pergunta = "Entrada de animais no Clube?"
resposta = qa_chain.invoke(pergunta)

# query_variacoes = MultiQueryRetriever.get_rephrased_queries(llm=llm, query=pergunta)
#
# print("\n🔁 Variações geradas da pergunta (RAG Fusion):\n")
# for i, q in enumerate(query_variacoes, start=1):
#     print(f"{i}. {q}")

# 9. Exibir resposta
print("\n🔍 Pergunta:", pergunta)
print("\n📘 Resposta:", resposta["result"])

print("\n📎 Fontes:")
for doc in resposta["source_documents"]:
    print(f"- {doc.metadata.get('source')}")
