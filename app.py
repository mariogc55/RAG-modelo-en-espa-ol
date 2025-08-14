import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
from deep_translator import GoogleTranslator


pdf_path = "libro de automatas.pdf" # Reemplaza con la ruta a tu PDF
model_path = "Lexi-Llama-3-8B-Uncensored_Q4_K_M.gguf" # Ajusta según la ubicación de tu modelo
persist_directory = "chroma_db"

def crear_base_de_conocimiento():
    # 1. Cargar el documento PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2. Fragmentar el texto
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # 3. Crear incrustaciones
    embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")

    # 4. Crear y persistir la base de datos vectorial Chroma
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
    db.persist()
    print("Base de conocimiento creada y persistida en chroma_db")


def consultar_base_de_conocimiento(query):

    # 0. Traducir la pregunta del usuario de español a inglés
    try:
        translated_query = GoogleTranslator(source='es', target='en').translate(query)
        print(f"Pregunta traducida (a inglés): {translated_query}")
    except Exception as e:
        print(f"Error al traducir la pregunta: {e}")
        translated_query = query # Si la traducción falla, usa la pregunta original

    # 1. Cargar las incrustaciones
    embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    # 2. Inicializar el modelo LlamaCpp
    llm = LlamaCpp(
        model_path = model_path,
        model_kwargs={"n_gpu_layers": 1},
        n_batch=512,
        n_ctx=8192,
        callback_manager=None,
        verbose=False,
    )

    # 3. Crear la cadena de recuperación y ejecutar la consulta con la pregunta traducida
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())
    result_en = qa.run(translated_query)

    # 5. Traducir la respuesta de vuelta a español
    try:
        result_es = GoogleTranslator(source='en', target='es').translate(result_en)
    except Exception as e:
        print(f"Error al traducir la respuesta: {e}")
        result_es = result_en # Si la traducción falla, usa la respuesta en inglés

    # 6. Ejecutar la consulta
    print(f"Pregunta original: {query}")
    print(f"Respuesta (en español): {result_es}")

if __name__ == "__main__":
    if not os.path.exists(persist_directory):
        crear_base_de_conocimiento()
    else:
        print("La base de conocimiento ya existe. Omitiendo la creación.")

    while True:
        pregunta = input("Ingresa tu pregunta (o 'salir' para terminar): ")
        if pregunta.lower() == "salir":
            break
        consultar_base_de_conocimiento(pregunta)