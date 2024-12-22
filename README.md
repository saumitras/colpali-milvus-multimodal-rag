MultiModal Rag with Colpali and Milvus
===

Code for blog [https://saumitra.me/2024/2024-11-15-colpali-milvus-rag/](https://saumitra.me/2024/2024-11-15-colpali-milvus-rag/) on how to do multimodal RAG with [colpali](https://arxiv.org/abs/2407.01449), [milvus](https://milvus.io/) and a visual LLM (gemini/gpt-4o)

Demo running at [https://huggingface.co/spaces/saumitras/colpali-milvus](https://huggingface.co/spaces/saumitras/colpali-milvus)

Application will allow users to upload a PDF and then perform search or Q&A queries on both the text and visual elements of the document. We will not extract text from the PDF; instead, we will treat it as an image and use colpali to get embeddings for the PDF pages. These embeddings will be indexed to Milvus, and then we will use a visual LLM (gemini/gpt-4o) to facilitate the Q&A queries.
