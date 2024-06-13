from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
import openai


def responder_cie(Pregunta):

    ''' Initialize the QA model, prompt template, and chain '''

    system_promp_faiss = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n  if the user asks for the list of codes od diseases, give him the indexes\n\n
    
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """

    model_faiss = OpenAI(openai_api_key=openai.api_key, model_name="gpt-3.5-turbo-instruct", max_tokens="1000")
    prompt_pregunta = PromptTemplate(
        template=system_promp_faiss, input_variables=["input_documents", "question"]
    )
    chain = load_qa_chain(model_faiss, chain_type="stuff", prompt=prompt_pregunta)
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
    db = FAISS.load_local(
        folder_path="/teamspace/studios/this_studio/project/src/services/faiss_index",
        embeddings=embeddings,
        index_name="myFaissIndex",
        allow_dangerous_deserialization=True
    )

    docs = db.similarity_search(Pregunta)  
    response = chain({"input_documents": docs, "question": Pregunta})

    return response['output_text']
