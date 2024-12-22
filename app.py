import gradio as gr
import tempfile
import os
import fitz  # PyMuPDF
import uuid


from middleware import Middleware
from rag import Rag

rag = Rag()

def generate_uuid(state):
    # Check if UUID already exists in session state
    if state["user_uuid"] is None:
        # Generate a new UUID if not already set
        state["user_uuid"] = str(uuid.uuid4())

    return state["user_uuid"]


class PDFSearchApp:
    def __init__(self):
        self.indexed_docs = {}
        self.current_pdf = None
    
        
    def upload_and_convert(self, state, file, max_pages):
        id = generate_uuid(state)

        if file is None:
            return "No file uploaded"

        print(f"Uploading file: {file.name}, id: {id}")
            
        try:
            self.current_pdf = file.name

            middleware = Middleware(id, create_collection=True)

            pages = middleware.index(pdf_path=file.name, id=id, max_pages=max_pages)

            self.indexed_docs[id] = True
            
            return f"Uploaded and extracted {len(pages)} pages"
        except Exception as e:
            return f"Error processing PDF: {str(e)}"
    
    
    def search_documents(self, state, query, num_results=1):
        print(f"Searching for query: {query}")
        id = generate_uuid(state)
        
        if not self.indexed_docs[id]:
            print("Please index documents first")
            return "Please index documents first", "--"
        if not query:
            print("Please enter a search query")
            return "Please enter a search query", "--"
            
        try:

            middleware = Middleware(id, create_collection=False)
            
            search_results = middleware.search([query])[0]

            page_num = search_results[0][1] + 1

            print(f"Retrieved page number: {page_num}")

            img_path = f"pages/{id}/page_{page_num}.png"

            print(f"Retrieved image path: {img_path}")

            rag_response = rag.get_answer_from_gemini(query, [img_path])

            return img_path, rag_response
            
        except Exception as e:
            return f"Error during search: {str(e)}", "--"

def create_ui():
    app = PDFSearchApp()
    
    with gr.Blocks() as demo:
        state = gr.State(value={"user_uuid": None})

        gr.Markdown("# Colpali Milvus Multimodal RAG Demo")
        gr.Markdown("This demo showcases how to use [Colpali](https://github.com/illuin-tech/colpali) embeddings with [Milvus](https://milvus.io/) and utilizing Gemini/OpenAI multimodal RAG for pdf search and Q&A.")
        
        with gr.Tab("Upload PDF"):
            with gr.Column():
                file_input = gr.File(label="Upload PDF")
                
                max_pages_input = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=20,
                    step=10,
                    label="Max pages to extract and index"
                )
                
                status = gr.Textbox(label="Indexing Status", interactive=False)
        
        with gr.Tab("Query"):
            with gr.Column():
                query_input = gr.Textbox(label="Enter query")
                # num_results = gr.Slider(
                #     minimum=1,
                #     maximum=10,
                #     value=5,
                #     step=1,
                #     label="Number of results"
                # )
                search_btn = gr.Button("Query")
                llm_answer = gr.Textbox(label="RAG Response", interactive=False)
                images = gr.Image(label="Top page matching query")
        
        # Event handlers
        file_input.change(
            fn=app.upload_and_convert,
            inputs=[state, file_input, max_pages_input],
            outputs=[status]
        )
        
        search_btn.click(
            fn=app.search_documents,
            inputs=[state, query_input],
            outputs=[images, llm_answer]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch()
