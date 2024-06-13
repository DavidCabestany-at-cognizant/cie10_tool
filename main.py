from src.api.gradio_interface import create_interface
from src.utils.gpt_utils import load_environment, initialize_model_and_chain

def main():
    config = load_environment()
    embeddings, chain = initialize_model_and_chain(config)
    interface = create_interface(config, chain, embeddings)
    interface.launch(share=True)

if __name__ == "__main__":
    main()

