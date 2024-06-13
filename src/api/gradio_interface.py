import gradio as gr
from ..services.processing_service import process_and_combine_images
from ..utils.file_utils import save_to_json, save_to_excel
from ..utils.image_utils import convert_pdf_to_images
from ..utils.qa_utils import responder_cie

def create_interface(config, chain, embeddings):
    css = """
    #fixed-pdf-input {
        width: 100%;
        height: 300px;
        overflow: hidden;
        border: 1px solid #ccc;
        padding: 10px;
        box-sizing: border-box;
    }
    """
    with gr.Blocks(css=css, theme=gr.themes.Default(primary_hue=gr.themes.colors.indigo).set(button_primary_background_fill="*primary_400")) as interface:
        with gr.Tab("Procesador PDF"):
            gr.Markdown(
                """
                # Extractor de Datos desde Archivos PDF
                Cargue un PDF para extraer e identificar las casillas marcadas mediante OCR y GPT-4. 
                Haga clic en «Enviar» para extraer los campos y en «Mostrar Imágenes» para mostrar las páginas PDF.
                """
            )
            with gr.Row():
                with gr.Column():
                    pdf_input = gr.File(type="filepath", label="Cargar PDF", elem_id="fixed-pdf-input")
                    orientation = gr.Dropdown(["vertical", "horizontal"], label="Orientación", value="horizontal")
                    submit_button = gr.Button("Enviar", variant="primary")
                    clear_button = gr.Button("Limpiar")
                with gr.Column():
                    text_output = gr.Textbox(label="Campos Extraídos", placeholder="La información extraída aparecerá aquí...", lines=23)
            image_gallery = gr.Gallery(label="Páginas PDF", visible=True)
            show_images_button = gr.Button("Mostrar Imágenes", visible=True)
            gr.Markdown(
                """
                # Asistente virtual
                Compruebe los códigos CIE-10 y devuelve la última versión publicada por la OMS
                """
            )
            with gr.Row():
                inputs = gr.Textbox(label="Pregunta", lines=5)
                outputs = gr.Textbox(label="Respuesta", lines=5)
            with gr.Row():
                enviar_button = gr.Button("Enviar", variant="primary")
                clear_button2 = gr.Button("Limpiar")
            enviar_button.click(fn=responder_cie, inputs=[inputs], outputs=[outputs])
            clear_button2.click(fn=lambda: (None, gr.update(visible=True), gr.update(visible=True)), outputs=[outputs])
            with gr.Row():
                with gr.Column():
                    download_json_button = gr.Button("Descargar en JSON", visible=True)
                    download_excel_button = gr.Button("Descargar en Excel", visible=True)
            model_state = gr.State(config["model"])
            temperature_state = gr.State(config["temperature"])
            max_tokens_state = gr.State(config["max_tokens"])
            top_p_state = gr.State(config["top_p"])
            frequency_penalty_state = gr.State(config["frequency_penalty"])
            presence_penalty_state = gr.State(config["presence_penalty"])
            submit_button.click(
                fn=lambda pdf_file_path, orientation: process_and_combine_images(pdf_file_path, orientation, config), 
                inputs=[pdf_input, orientation], outputs=[text_output, image_gallery]
            )
            show_images_button.click(fn=convert_pdf_to_images, inputs=pdf_input, outputs=image_gallery)
            clear_button.click(fn=lambda: (None, gr.update(visible=True), gr.update(visible=True)), outputs=[text_output, show_images_button, image_gallery])
            download_json_button.click(fn=save_to_json, inputs=[text_output], outputs=[gr.File()])
            download_excel_button.click(fn=save_to_excel, inputs=[text_output, pdf_input], outputs=[gr.File()])

        with gr.Tab("Configuration"):
            gr.Markdown(
                """
                # Configuration Settings
                Adjust the GPT-4 parameters to fine-tune the extraction process.
                """
            )
            model = gr.Textbox(value=config["model"], label="Model")
            temperature = gr.Slider(0.0, 1.0, value=config["temperature"], step=0.1, label="Temperature")
            max_tokens = gr.Slider(10, 4000, value=config["max_tokens"], step=50, label="Max Tokens")
            top_p = gr.Slider(0.0, 1.0, value=config["top_p"], step=0.1, label="Top P")
            frequency_penalty = gr.Slider(-2.0, 2.0, value=config["frequency_penalty"], step=0.1, label="Frequency Penalty")
            presence_penalty = gr.Slider(-2.0, 2.0, value=config["presence_penalty"], step=0.1, label="Presence Penalty")
            save_button = gr.Button("Save Settings")
            save_button.click(
                fn=lambda model, temperature, max_tokens, top_p, frequency_penalty, presence_penalty: (
                    model, temperature, max_tokens, top_p, frequency_penalty, presence_penalty),
                inputs=[model, temperature, max_tokens, top_p, frequency_penalty, presence_penalty],
                outputs=[model_state, temperature_state, max_tokens_state, top_p_state, frequency_penalty_state, presence_penalty_state]
            )
    return interface
