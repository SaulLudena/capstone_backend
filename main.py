import logging
import os

import fitz  # PyMuPDF
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from io import BytesIO
from openai import OpenAI
app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"
client = OpenAI(api_key="API_KEY_SECRETA")
# Configuración básica del logging
logging.basicConfig(
    level=logging.DEBUG,  # Nivel de logging
    format="%(asctime)s [%(levelname)s] %(message)s",  # Formato de los mensajes de log
    handlers=[
        logging.FileHandler("app.log"),  # Archivo donde se guardarán los logs
        logging.StreamHandler(),  # También se mostrará en la consola
    ],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def pdf_to_text(pdf_path):
    text = ""
    try:
        document = fitz.open(pdf_path)
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text += page.get_text("text")
    except Exception as e:
        logging.error(f"Error extracting text from PDF '{pdf_path}': {str(e)}")
    return text

@app.route("/", methods=["POST"])
@cross_origin()
def root():
    if "pdfFiles" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    pdf_files = request.files.getlist("pdfFiles")
    text_responses = []

    for pdf in pdf_files:
        pdf_path = os.path.join(UPLOAD_FOLDER, pdf.filename)
        pdf.save(pdf_path)
        text_from_pdf = pdf_to_text(pdf_path)  # Convert PDF to text
        text_responses.append(text_from_pdf)

    MODEL="gpt-4o"

    completion = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": 
         "Eres un investigador cientifico en ingeniería informática, debes cuidar tu redacción y entender el contenido de los documentos, puedes ayudarte de otras fuentes pero principalmente debes entender el contenido de los documentos."},
        {"role": "user", "content": "Genera el estado del arte de la siguiente informacion relacionada a una investigación cientifica, el estado del arte debe que generes contar con 400 palabras en un único parrafo, deberás incluir datos cuantitativos y deberá contar con estos puntos escenciales e importantes: Actualización, Relevancia, Exhaustividad, Organizacion y claridad, Comparacion y contraste e indentificacion de desafios y problemas abiertos. El documento fuente es:"+ text_from_pdf},

    ]
    )
    app.logger.info(completion.choices[0].message.content)

    return jsonify({"text_responses": text_responses,
                    "completion": completion.choices[0].message.content
                    }), 200

if __name__ == "__main__":
    app.run(debug=True)
