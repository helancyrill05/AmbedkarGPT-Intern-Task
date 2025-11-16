AmbedkarGPT - RAG-based Q\&A System



A Retrieval-Augmented Generation (RAG) system that answers questions based on Dr. B.R. Ambedkar's "Annihilation of Caste" speech.





1.Technologies Used:

\- Python 3.10

\- LangChain framework

\- ChromaDB (local vector store)

\- HuggingFace Embeddings (sentence-transformers/all-MiniLM-L6-v2)

\- Ollama with Mistral 7B





1\. Install Ollama

Download from: https://ollama.ai

Then run in terminal:

ollama pull mistral



2\. Clone Repository

```bash

git clone https://github.com/helancyrill05/AmbedkarGPT-Intern-Task.git

cd AmbedkarGPT-Intern-Task

```



3\. Setup Environment and install dependencies

```bash

conda create -n rag-env python=3.10

conda activate rag-env



pip install -r requirements.txt

```



4\. Run

```bash

python main.py

```



After running, the system will:

1\. Load the speech text

2\. Create embeddings (first time: ~1 minute)

3\. Answer 3 example questions

4\. Open interactive mode



Try asking:

\- "What is the main argument?"

\- "What solutions are proposed?"

\- "Summarize in 3 sentences"



Commands:

\- `history` - View past questions

\- `save` - Export Q\&A to JSON

\- `stats` - System info

\- `quit` - Exit



Project Structure



AmbedkarGPT-Intern-Task/

├── main.py              # Main application code

├── requirements.txt     # Python dependencies

├── README.md           # This file

└── data/

&nbsp;   └── speech.txt      # Ambedkar's speech





Author: \*\*Helan Cyrill\*\*  

GitHub: \[@helancyrill05](https://github.com/helancyrill05)

