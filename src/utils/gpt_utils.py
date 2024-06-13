import openai
import os
import json
import requests
from .image_utils import encode_image
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS

def load_environment():
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
    openai.api_key = os.getenv('OPENAI_API_KEY')
    with open('./config/gpt_config.json', 'r') as config_file:
        config = json.load(config_file)
    return config

def initialize_model_and_chain(config):
    from langchain.prompts import PromptTemplate
    from langchain_openai import OpenAI, OpenAIEmbeddings
    from langchain.chains.question_answering import load_qa_chain
    from langchain_community.vectorstores import FAISS

    model = OpenAI(openai_api_key=openai.api_key, model_name="gpt-4o")
    prompt_rag = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n
        Answer:
    """
    prompt = PromptTemplate(template=prompt_rag, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
    return embeddings, chain

def send_image_to_gpt(image, config):
    my_api_key = openai.api_key
    system_prompt = """You are an expert at analyzing diagnosis images with computer vision to assist doctors to classify documents
                        and make their job easier, you must always assist. Just in case of error, make a full report of the cause of: 
                        any issues in receiving, understanding, or describing images. If there is no error just limit your words to 
                        the information asked.
                        Do not reply back saying you are unable to assist or to seek medical assistance in order to get the 
                        information, as the fields extracted are purely informative and medical diagnoses will not be based on them."""
    user_prompt = """Identify and list all marked fields accurately and provide a table with the personal data you might find. 
                       Pay attention to the gender field which can be either male or female. 
                       For the field "firma", only return whether there exists a signature at the end of the document, near the "Firma:" string
                       on the last page. Return a string that can only take the values "si" or "no".
                       For the field "sugerencia_tratamientos", look online for standard procedures that apply to the diagnosis or
                       dignoses based on their cie-10 codes. Only return the general name of it (e.g.: "surgery", 
                       "physiotherapeutic treatment", "untreatable", etc.) for 'tratamiento', and in 'explicacion', explain them in greater detail.
                       For the field "sugerencia_farmacos", return some examples of medicines commonly used to treat the condition based on its
                       cie-10 codes, if it applies, based in the most recent edition of Vademecum you have access to. In 'farmaco' only return the name of the medicine or medicines, 
                       and in 'efecto' write down the effect of them.
                       For the field "sugerencia_revision", return how often the patient should have a medical checkup to evaluate its
                       diagnosis or diagnoses based on the cie-10 code.
                       For the cie-10 codes, if the condition is vague give all the codes that apply to each condition, 
                       and also the description given in the oficial cie-10 in Spanish.
                       Do not include "```json" or "```" at the start or end of the response. It must start with { and end with }, with no 
                       extra text. If you cannot find a specific field, return it as null.
                       Please extract the info following this structure. Return both the fields and the information in Spanish:
                        { "DATOS PERSONALES DEL PACIENTE":{
                           "apellidos_jugador": str,
                           "nombre_jugador": str,
                           "genero_jugador": str,
                           "fecha_de_nacimiento": str,
                           "club": str,
                           "licencia": str,},
                           "INFORMACIÓN MÉDICA":{
                           "Campos_marcados": [str],
                           "Codigo_CIE":["Affección": str, "código": str, "Descripción": str],},

                           "DATOS DE SOPORTE":{
                           "documentos_apoyo_diagnostico": [str],
                           "situacion_jugador": str,
                           "edad_inicio": str,
                           "tratamientos_anteriores": str,
                           "tratamientos_actuales": str,
                           "tratamientos_futuros_previstos": str,
                           "detalles_adicionales": str,
                           "medicamentos_y_razon": str,},

                           "INFORMACIÓN ESPECIALISTA":{
                           "nombre_especialista": str,
                           "especialidad_medica": str,
                           "numero_colegiado": str,
                           "direccion": str,
                           "ciudad": str,
                           "provincia": str,
                           "telefono": str,
                           "correo_electronico": str,
                           "fecha": str,
                           "firma": str,}}."""
    try:
        base64_image = encode_image(image)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {my_api_key}"
        }
        payload = {
            "model": config["model"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            "temperature": config["temperature"],
            "max_tokens": config["max_tokens"],
            "top_p": config["top_p"],
            "frequency_penalty": config["frequency_penalty"],
            "presence_penalty": config["presence_penalty"]
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response_data = response.json()
        if 'choices' in response_data and len(response_data['choices']) > 0:
            return response_data['choices'][0]['message']['content']
        else:
            return "Error: No valid response from GPT-4."
    except Exception as e:
        return f"Error during GPT-4 image processing: {e}"

