---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region tags=["title"] -->
# Large Language Models as Tools for Historical Research
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} tags=["contributor"] -->
### Jacob Forward [![orcid](https://orcid.org/sites/default/files/images/orcid_16x16.png)](https://orcid.org/0009-0008-7320-2240) 
University of Cambridge
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} tags=["contributor"] -->
### Ryan Daniels [![orcid](https://orcid.org/sites/default/files/images/orcid_16x16.png)](https://orcid.org/0000-0003-4805-1598) 
University of Cambridge
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} tags=["copyright"] -->
[![cc-by](https://licensebuttons.net/l/by/4.0/88x31.png)](https://creativecommons.org/licenses/by/4.0/) 
©<Jacob Forward, Ryan Daniels>. Published by De Gruyter in cooperation with the University of Luxembourg Centre for Contemporary and Digital History. This is an Open Access article distributed under the terms of the [Creative Commons Attribution License CC-BY](https://creativecommons.org/licenses/by/4.0/)

<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""} tags=["cover"]
from IPython.display import Image, display

display(Image("./media/rvr_title_pic.png", width=750))
```

<!-- #region tags=["keywords"] -->
Large Language Models, Retrieval Augmented Generation, Finetuning, American History, US Politics
<!-- #endregion -->

<!-- #region tags=["abstract"] -->
This article investigates the transformative potential of Large Language Models (LLMs) in historical research, proposing a dual methodological framework to navigate the "age of abundance" in digital archives. We contrast Retrieval Augmented Generation (RAG) with fine-tuning to demonstrate how LLMs serve distinct epistemological functions: as semantic search engines and as simulators of historical discourse. Using corpora of American Presidential speeches, our RAG system illustrates the power of vector-based retrieval to surface relevant content without relying on exact keyword matches, thereby enhancing archival discovery while maintaining clear source provenance. Conversely, by fine-tuning models on presidential Q&A datasets, we show how LLMs can internalise specific historical voices, offering a novel mechanism to study discourse through generative simulation. We argue these approaches are complementary, where RAG prioritises the verifiable retrieval of information, fine-tuning offers a hermeneutic tool for modelling the probabilistic logic of historical language. Ultimately, this study posits that the emergence of LLMs necessitates a disciplinary pivot, in which historians will increasingly blend traditional close reading with computational "distant reading" to better interrogate the vast, rapidly accumulating, unstructured data of the past.
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} -->
## Introduction
<!-- #endregion -->

<!-- #region citation-manager={"citations": {"2na64": [{"id": "22606145/M8V7J2QU", "source": "zotero"}], "30nin": [{"id": "22606145/Z75YKV7A", "source": "zotero"}], "fk113": [{"id": "22606145/WUKNYC5F", "source": "zotero"}], "iyqvc": [{"id": "22606145/XG5LV6XR", "source": "zotero"}], "kp0tu": [{"id": "22606145/IBKLEB8N", "source": "zotero"}]}} editable=true slideshow={"slide_type": ""} -->
Source abundance is perhaps the biggest methodological challenge facing most historians today. As early as the 1960s, E. H. Carr wrote that scholars of ancient and medieval history ought to be grateful for “the vast winnowing process” of source survival over time which had endowed them with “a manageable corpus” of sources, as opposed to modern historians who must cultivate a “necessary ignorance” for themselves or else be incapacitated by the sheer volume of potentially significant evidence <cite id="2na64"><a href="#zotero%7C22606145%2FM8V7J2QU">(Carr, 1964)</a></cite>. However, the ephemerality of sources in the digital age that exist as mere magnetic impulses on distant servers led Rosenzweig in 2003 to envision two futures: either scarcity because of the fragility of electronic records or abundance <cite id="fk113"><a href="#zotero%7C22606145%2FWUKNYC5F">(Rosenzweig, 2003)</a></cite>. By 2008 it had become clear that “historians (would) have to grapple with abundance, not scarcity,” which posed the new problem of the “infinite archive,” <cite id="iyqvc"><a href="#zotero%7C22606145%2FXG5LV6XR">(“Interchange: The Promise of Digital History,” 2008)</a></cite>. Nor was this problem confined to historians studying the recent past, since increasing numbers of historians who worked with traditional archives started to photograph vast numbers of documents, and libraries and archives rushed to digitise their newspaper and image collections <cite id="30nin"><a href="#zotero%7C22606145%2FZ75YKV7A">(Milligan, 2019)</a></cite>. The haystack of information hiding the few potentially salient needles of evidence has grown and continues to grow at an exponential rate, and with it, the risk to historians of confirmation bias. To adapt Kuny’s famous phrase, it seems it is abundance, not scarcity, that risks imposing a digital Dark Age <cite id="kp0tu"><a href="#zotero%7C22606145%2FIBKLEB8N">(Terry Kuny, 1997)</a></cite>.
<!-- #endregion -->

<!-- #region citation-manager={"citations": {"ch1sz": [{"id": "22606145/RTA8HXQ8", "source": "zotero"}], "gv1lr": [{"id": "22606145/WUKNYC5F", "source": "zotero"}], "k7xuo": [{"id": "22606145/SQUX3KFK", "source": "zotero"}]}} -->
The accounts of, for instance, contemporary historians writing on American politics assemble formidable arsenals of facts and figures and deploy them with truly compelling eloquence to advance their arguments. Yet, no matter how comprehensive and definitive their accounts may sound, they cannot claim to have engaged with a fraction of a percent of the potentially salient primary sources, nor that the sources they chose to engage with, often as a consequence of the idiosyncrasies of historical research including prior expertise, source visibility, or chance, are reliably representative of the whole. It is all too easy for facts to fit theories rather than for theories to be fashioned from the facts. Paul Boyer has referred to this as the “awesomely forbidding task” of writing contemporary American history, given the “dizzying abundance of potential sources” <cite id="ch1sz"><a href="#zotero%7C22606145%2FRTA8HXQ8">(Boyer, 2012)</a></cite>. By one illustrative estimate, the Clinton White House produced about 6 million e-mails per year towards the end of his second term in office, and the pronounced volume increase is similar in presidential spoken addresses and remarks <cite id="gv1lr"><a href="#zotero%7C22606145%2FWUKNYC5F">(Rosenzweig, 2003)</a></cite>. In Carla Hesse’s view, this is part of the reason why so many historians have jumped ship from social to cultural history, from the older “plodding work of looking and counting” to the sheer joy of reading and interpreting freely that accompanied the linguistic turn <cite id="k7xuo"><a href="#zotero%7C22606145%2FSQUX3KFK">(Hesse, 2004)</a></cite>. We contend that by enlisting Large Language Models (LLMs) to do this important ‘plodding work of looking and counting,’ historians can combine the rigour of a systematic and much more comprehensive source engagement with the joys of interpreting and weaving a narrative around the results.

<!-- #endregion -->

<!-- #region citation-manager={"citations": {"1hn04": [{"id": "22606145/NFQH6NBQ", "source": "zotero"}], "784jc": [{"id": "22606145/S4SBYSDZ", "source": "zotero"}], "gzqyk": [{"id": "22606145/4PNYFUFA", "source": "zotero"}], "jr90p": [{"id": "22606145/H7LI7SEM", "source": "zotero"}], "omqvq": [{"id": "22606145/5CXSNZLL", "source": "zotero"}], "zmz86": [{"id": "22606145/H7LI7SEM", "source": "zotero"}]}} editable=true slideshow={"slide_type": ""} -->
Since the rapid popularization of LLMs with ChatGPT in 2022-3, there have been growing calls for historians and humanists to play a role in assuring a critical use of AI tools, or risk “letting algorithms shape our research in opaque and unforeseen ways” <cite id="zmz86"><a href="#zotero%7C22606145%2FH7LI7SEM">(Schmidt, 2023)</a></cite>. The American Historical Association recently acknowledged that as historians “our fundamental association with evidence, bias, historical method, and context, are all in flux,” as a direct consequence of the emergence of powerful LLMs on the scene <cite id="jr90p"><a href="#zotero%7C22606145%2FH7LI7SEM">(Schmidt, 2023)</a></cite>. There are only a few early examples of scholars deploying LLMs for historical research. As part of a multidisciplinary research project based in the UK called ‘Living with Machines’, a team fine-tuned a few versions of the BERT language model on a corpus of 19th-century English literature to facilitate research into contemporary language use. The paper is rather technical and focuses on presenting the training of the models rather than any downstream use cases, but it remains one of the few and perhaps the earliest example of an explicit recognition of the value of finetuning LLMs “for historical research” as they facilitate “a more accurate context dependent representation of meaning" <cite id="1hn04"><a href="#zotero%7C22606145%2FNFQH6NBQ">(Kasra Hosseini et al., 2021)</a></cite>. In 2022, we see the first mention of LLMs by the American Sociological Society in a journal article introducing LLMs “to a general social science audience,” as a method of the future with high potential <cite id="784jc"><a href="#zotero%7C22606145%2FS4SBYSDZ">(Jensen et al., 2022)</a></cite>. This was followed in October 2022 with the first published attempt to use finetuned LLMs for historical purposes in Chantalle Brousseau’s excellent lesson on Programming Historian which walked users through finetuning a small version of GPT2 on a corpus of press coverage on Brexit, and then prompting the finetuned models <cite id="omqvq"><a href="#zotero%7C22606145%2F5CXSNZLL">(Chantal Brousseau, 2022)</a></cite>. At the same time as Brousseau published this lesson, another project was working on the application of LLMs for a discourse analysis task, fine-tuning a model on a corpus of State of the Union addresses <cite id="gzqyk"><a href="#zotero%7C22606145%2F4PNYFUFA">(Forward, 2023)</a></cite>. We can therefore situate our methods within a nascent and rapidly growing community of researchers in many fields.

<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} -->
In this article, we will share the findings of a year-long research project on experimental applications of LLMs to historical research workflows, especially how LLMs can help tackle the source abundance problem. The project’s case study focuses on two large corpora (totalling nearly 6 million words) of spoken addresses and remarks from presidents Franklin D. Roosevelt and Ronald Reagan, digitised by the [American Presidency Project](https://www.presidency.ucsb.edu/). These figures are chosen as icons of the New Deal and Neoliberalism, respectively; two forces in US politics that have profoundly shaped the global order in the 20th century. We have principally experimented with two methods. The first is the use of retrieval augmented generation (RAG) techniques for embedding, searching, and retrieving presidential speeches, and compiling them into historically accurate generated answers. The second is the fine-tuning of small LLMs on question and answer pairs from presidential press conferences.

<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} -->
## Retrieval Augmented Generation: Sources and Method
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} -->
Why build a Retrieval Augmented Generation (RAG) system? RAG offers a profound advantage over traditional keyword search when it comes to tackling the source abundance problem by bridging the gap between rigid data retrieval and semantic understanding. As illustrated in the Source Document Ingestion phase in the diagram below, RAG does not simply index words; it uses an Embedding Model to convert the data, in this case our Presidential Speech Corpora, into multidimensional vectors stored in a Vector Database. This allows for semantic search, where the system identifies relevant historical context based on conceptual meaning rather than exact phrasing. Furthermore, while traditional word search engines leave the user to synthesise findings themselves, the LLM in this pipeline ingests the specific Retrieved Context to construct a coherent Generated Text Response. This transforms the research process from a manual "hunt-and-peck" through archives into a dynamic dialogue where the researcher can interrogate the primary sources directly using full 'human-language' questions, as opposed to the 'machine-language' of [regex](https://docs.python.org/3/library/re.html) or advanced keyword search methods.
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""} tags=["figure-rag_diagram-*"]
from IPython.display import Image, display

display(Image("./media/rag_diagram.jpg", width=1000))
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
We began by defining two corpora, one for FDR and one for Reagan, by using the advanced search parameters on the American Presidency Project website to isolate sources tagged as 'spoken addresses and remarks', and dating to within their respective terms in office. We used these search result URLs as the starting point for a server-respectful and compliant scrape of the search results. For more details on our scraping method, cleaning steps, and the full resulting dataset, please see our [Project Github Page](https://github.com/JacobF99/speaking_with_our_sources/blob/main/data_README.md). Those wishing to replicate our research are invited to start with this carefully prepared dataset and the embedding and retrieval methods discussed below. 
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} tags=["hermeneutics"] -->
The first step in creating a retrieval augmented generation pipeline is vectorising chunks of text from the corpus (transforming them into high-dimensional numerical vectors that represent the semantic contents of each chunk of text). We spent quite some time finessing the right 'chunk size' to embed in this process. Too small a chunk of words, and there is not enough context to make it useful when retrieved. Too large, and the vector search process is hampered, and the accuracy of pinpointing specific information is sacrificed a little. A fine balance needs to be struck between enough context and enough granularity to make the search process precise and the retrieved content rich. In our case, given the average length of speeches, a chunk size of 150 words emerged as a rough optimum for our purposes. The following script is an example of how to vectorise text using [an open-source sentence embedding model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) from Huggingface. In this example, we use the popular vector database storage service [Pinecone](https://www.pinecone.io/), which is proprietary but has an ample free tier suitable for research purposes. 
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""} tags=["hermeneutics"]
# ==========================================
# 1. Setup and Dependencies
# ==========================================

# I would advise selecting a T4-GPU runtime type if running this in Google Colab
# You will also need a Pinecone account and API key
# Install necessary libraries if not present
# !pip install pandas numpy sentence-transformers pinecone-client nltk tqdm

import os
import pandas as pd
import numpy as np
import nltk
from typing import List, Dict, Any
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from google.colab import drive

# Download necessary NLTK data for sentence tokenisation
nltk.download('punkt', quiet=True)

# ==========================================
# 2. Configuration
# ==========================================

CONFIG = {
    # File System Paths
    "drive_mount_path": "/content/drive",
    "source_folder": "/content/drive/MyDrive/Your_Source_Folder", # UPDATE THIS PATH
    
    # Embedding Settings
    "model_name": "all-MiniLM-L6-v2",
    "chunk_target_words": 150,
    
    # Pinecone Settings
    "index_name": "fdr-academic-rag",
    "metric": "cosine",
    "dimension": 384, # Dimensions must match the model_name (384 for MiniLM)
    "cloud": "aws",
    "region": "us-east-1", # UPDATE IF DIFFERENT
    "batch_size": 100 # Number of vectors to upsert at once
}

# ==========================================
# 3. Helper Functions: Processing & Logic
# ==========================================

def mount_drive():
    """Mounts Google Drive to the Colab environment."""
    if not os.path.exists(CONFIG["drive_mount_path"]):
        drive.mount(CONFIG["drive_mount_path"])
        print("Google Drive mounted successfully.")
    else:
        print("Drive already mounted.")

def chunk_text(text: str, target_words: int = 150) -> List[str]:
    """
    Splits text into chunks of approximately 'target_words' length.
    Preserves sentence boundaries using NLTK.
    """
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        sentence_word_count = len(sentence.split())
        # If adding this sentence exceeds target, finalise current chunk
        if current_word_count + sentence_word_count > target_words and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_word_count = sentence_word_count
        else:
            current_chunk.append(sentence)
            current_word_count += sentence_word_count

    # Append any remaining text
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def process_csv_file(file_path: str, model: SentenceTransformer) -> pd.DataFrame:
    """
    Reads a CSV, chunks the content, and generates embeddings.
    Expected CSV columns: 'date', 'title', 'content'
    """
    try:
        df = pd.read_csv(file_path)
        # Ensure standard column naming
        df.columns = [c.lower() for c in df.columns] 
        required_cols = ['date', 'title', 'content']
        
        if not all(col in df.columns for col in required_cols):
            print(f"Skipping {os.path.basename(file_path)}: Missing required columns.")
            return pd.DataFrame()

        # Combine title and content for fuller context
        df['full_text'] = df['title'].fillna('') + " " + df['content'].fillna('')
        
        processed_records = []
        
        for _, row in df.iterrows():
            text_chunks = chunk_text(row['full_text'], CONFIG["chunk_target_words"])
            
            for chunk in text_chunks:
                processed_records.append({
                    'id_base': f"{row['title']}_{row['date']}", # used for vector ID generation
                    'date': str(row['date']),
                    'title': str(row['title']),
                    'chunk': chunk,
                    'source_file': os.path.basename(file_path)
                })

        if not processed_records:
            return pd.DataFrame()

        result_df = pd.DataFrame(processed_records)
        
        # Generate Embeddings
        # Note: encode returns numpy array, we convert to list for Pinecone
        embeddings = model.encode(result_df['chunk'].tolist(), show_progress_bar=False)
        result_df['values'] = embeddings.tolist()
        
        return result_df

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return pd.DataFrame()

# ==========================================
# 4. Main Execution Pipeline
# ==========================================

def main():
    # 1. Setup Environment
    mount_drive()
    
    # Check for API Key
    api_key = os.environ.get('PINECONE_API_KEY')
    if not api_key:
        api_key = input("Please enter your Pinecone API Key: ")
    
    # 2. Initialize Model
    print(f"Loading embedding model: {CONFIG['model_name']}...")
    model = SentenceTransformer(CONFIG['model_name'])
    device = "GPU" if hasattr(model, 'device') and model.device.type == 'cuda' else "CPU"
    print(f"Model loaded on {device}.")

    # 3. Process Data
    print("Scanning source folder for CSV files...")
    all_files = [f for f in os.listdir(CONFIG["source_folder"]) if f.endswith('.csv')]
    
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {CONFIG['source_folder']}")

    all_dataframes = []
    print(f"Processing {len(all_files)} files...")
    
    for filename in tqdm(all_files, desc="Vectorizing Files"):
        file_path = os.path.join(CONFIG["source_folder"], filename)
        df_processed = process_csv_file(file_path, model)
        if not df_processed.empty:
            all_dataframes.append(df_processed)

    if not all_dataframes:
        print("No valid data processed. Exiting.")
        return

    final_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"Total chunks created: {len(final_df)}")

    # 4. Pinecone Initialization
    pc = Pinecone(api_key=api_key)
    
    # Create index if it doesn't exist
    if CONFIG["index_name"] not in pc.list_indexes().names():
        print(f"Creating index: {CONFIG['index_name']}")
        pc.create_index(
            name=CONFIG["index_name"],
            dimension=CONFIG["dimension"],
            metric=CONFIG["metric"],
            spec=ServerlessSpec(cloud=CONFIG["cloud"], region=CONFIG["region"])
        )
    
    index = pc.Index(CONFIG["index_name"])

    # 5. Upsert to Pinecone
    print("Upserting vectors to Pinecone...")
    
    # Prepare batch iterator
    total_upserted = 0
    
    for i in tqdm(range(0, len(final_df), CONFIG["batch_size"]), desc="Uploading Batches"):
        batch = final_df.iloc[i:i + CONFIG['batch_size']]
        
        vectors_to_upsert = []
        for idx, row in batch.iterrows():
            # Create a unique ID for the vector
            vector_id = f"{row['id_base']}_chunk_{idx}"
            
            # Prepare metadata (Pinecone metadata must be primitive types or lists of strings)
            metadata = {
                "date": row['date'],
                "title": row['title'],
                "chunk": row['chunk'], # This is the context text we retrieve later
                "source": row['source_file']
            }
            
            vectors_to_upsert.append((vector_id, row['values'], metadata))
        
        try:
            index.upsert(vectors=vectors_to_upsert)
            total_upserted += len(vectors_to_upsert)
        except Exception as e:
            print(f"Error upserting batch starting at index {i}: {e}")

    # 6. Final Stats
    stats = index.describe_index_stats()
    print("\nPipeline Complete.")
    print(f"Total vectors upserted this run: {total_upserted}")
    print(f"Total vectors currently in index: {stats.total_vector_count}")

if __name__ == "__main__":
    main()
```

<!-- #region editable=true slideshow={"slide_type": ""} tags=["hermeneutics"] -->
Once the vector database has been created and uploaded to Pinecone, it is possible to complete the RAG pipeline by querying the database using [an open-source LLM](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) from Huggingface, and a simple Gradio UI. In this stage, the user's question is vectorised, using the same embedding model that was used to vectorise the dataset. The question vector is then compared to the other vectors in the database using 'cosine similarity', and the most similar database vectors are returned and re-translated into human language. These translated chunks and the original question are then passed to the LLM, which uses this accurate context to respond to the question. Results with smaller open-source LLMs hosted for free in Google Colab can vary (due to the constraints on the size of the model one can load). The chunk retrieval step on its own may be of interest to historians looking for a richer search experience, even without the LLM to synthesise a response. 
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""} tags=["hermeneutics"]
# ==========================================
# 1. Setup and Dependencies
# ==========================================

# Install necessary libraries if not present
# You will again need your Pinecone API key and a free Huggingface Hub account, and a token
# !pip install gradio pinecone-client sentence-transformers torch transformers huggingface_hub accelerate

import os
import torch
import gradio as gr
from typing import Tuple, List, Dict
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, Pipeline

# ==========================================
# 2. Configuration
# ==========================================

CONFIG = {
    # Pinecone Settings (Must match your Ingestion Script)
    "index_name": "fdr-academic-rag",
    "embedding_model": "all-MiniLM-L6-v2", # Must match the model used for ingestion
    
    # LLM Settings
    # specific model from your notebook: Qwen 1.5B (Good balance of speed/quality for Colab T4 GPU)
    "llm_model_id": "Qwen/Qwen2.5-1.5B-Instruct", 
    "max_new_tokens": 512,
    "context_window": 2048,
    
    # System Prompt
    "system_persona": (
        "As President Franklin Delano Roosevelt, answer the following question based purely on the given context. "
        "Compile your answer using verbatim quotes from the returned sources as far as possible. 
        "If you cannot answer the question with the provided context, "
        "say 'I'm afraid I don't have sufficient information to address that matter at present.'"
    )
}

# ==========================================
# 3. Model Initialization
# ==========================================

class RAGPipeline:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing RAG Pipeline on {self.device}...")
        
        # 1. Load Embedding Model (for retrieval)
        print(f"Loading embedding model: {CONFIG['embedding_model']}...")
        self.embedder = SentenceTransformer(CONFIG['embedding_model'], device=self.device)
        
        # 2. Connect to Pinecone
        api_key = os.environ.get('PINECONE_API_KEY')
        if not api_key:
            api_key = input("Enter Pinecone API Key: ")
        
        pc = Pinecone(api_key=api_key)
        if CONFIG['index_name'] not in pc.list_indexes().names():
            raise ValueError(f"Index '{CONFIG['index_name']}' not found. Please run the ingestion script first.")
            
        self.index = pc.Index(CONFIG['index_name'])
        print("Connected to Pinecone.")

        # 3. Load LLM (for generation)
        # Note: You need a Hugging Face token defined in the environment or logged in via CLI
        print(f"Loading LLM: {CONFIG['llm_model_id']}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(CONFIG['llm_model_id'])
            self.llm = AutoModelForCausalLM.from_pretrained(
                CONFIG['llm_model_id'],
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
        except Exception as e:
            print(f"Error loading LLM: {e}. Ensure you have authenticated with Hugging Face.")
            raise

    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[str], str]:
        """
        Embeds the query and retrieves semantic matches from Pinecone.
        Returns: List of formatted chunks, and a single concatenated string of context.
        """
        # Embed query
        query_embedding = self.embedder.encode(query, convert_to_tensor=False).tolist()
        
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        formatted_sources = []
        raw_context_list = []
        
        for match in results['matches']:
            meta = match['metadata']
            # Format for display in UI
            source_display = (
                f"Date: {meta.get('date', 'Unknown')}\n"
                f"Title: {meta.get('title', 'Unknown')}\n"
                f"Excerpt: {meta.get('chunk', '')}\n"
            )
            formatted_sources.append(source_display)
            raw_context_list.append(meta.get('chunk', ''))
            
        return formatted_sources, "\n\n".join(raw_context_list)

    def generate(self, query: str, context: str, temperature: float = 0.7) -> str:
        """
        Constructs the prompt and generates the response using the LLM.
        """
        prompt_template = f"""{CONFIG['system_persona']}

Context:
{context}

Question: {query}

Answer:"""

        inputs = self.tokenizer(prompt_template, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=CONFIG['max_new_tokens'],
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        # Decode and strip the prompt to get just the answer
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Simple parsing to remove the prompt part (heuristic based on 'Answer:')
        if "Answer:" in full_response:
            response_text = full_response.split("Answer:")[-1].strip()
        else:
            response_text = full_response
            
        return response_text

# ==========================================
# 4. Gradio Interface Logic
# ==========================================

def launch_interface():
    # Initialise Pipeline
    rag = RAGPipeline()

    def chat_function(message, history, top_k, temperature):
        # 1. Retrieve
        sources_list, concatenated_context = rag.retrieve(message, top_k=int(top_k))
        
        # 2. Generate
        response = rag.generate(message, concatenated_context, temperature=float(temperature))
        
        # 3. Format Sources for display
        sources_display = "\n---\n".join(sources_list)
        
        return response, sources_display

    # Define UI Layout
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎙️ Presidential Archives: Chat with FDR")
        gr.Markdown(
            "This Retrieval Augmented Generation (RAG) system retrieves historical context "
            "from presidential speeches to generate authentic responses."
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.ChatInterface(
                    fn=chat_function,
                    additional_inputs=[
                        gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Retrieval Depth (Top-K)"),
                        gr.Slider(minimum=0.1, maximum=1.0, value=0.7, step=0.1, label="Creativity (Temperature)"),
                    ],
                    title="Ask the President",
                    description="Enter a question about the New Deal, WWII, or economic policy.",
                    examples=["What is your plan for the banking crisis?", "Why must we support the Allies?"],
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 🔍 Retrieved Historical Context")
                gr.Markdown("_The AI reads these specific excerpts to form its answer._")
                sources_output = gr.Textbox(
                    label="Source Excerpts", 
                    interactive=False, 
                    lines=20,
                    show_copy_button=True
                )
                
                # Setup custom event listener to update the source box
                # Note: Gradio's ChatInterface simplifies the chat, but to update a separate box
                # we hook into the submit event if we were building manually. 
                # Since ChatInterface is strict, we used 'additional_outputs' logic below:
                
    # *Correction for Gradio ChatInterface limitations*: 
    # ChatInterface expects the function to return only the message. 
    # To display sources side-by-side, we build a manual Chatbot interface below for full control.
    
    with gr.Blocks(theme=gr.themes.Soft()) as manual_demo:
        gr.Markdown("# 🎙️ Presidential Archives: Chat with FDR")
        
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(height=500, label="Conversation")
                msg = gr.Textbox(label="Your Question", placeholder="Ask about the state of the union...")
                
                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary")
                    clear_btn = gr.Button("Clear History")
                
                with gr.Accordion("Advanced Settings", open=False):
                    temp_slider = gr.Slider(0.1, 1.0, 0.7, label="Temperature")
                    k_slider = gr.Slider(1, 10, 5, 1, label="Top-K Sources")

            with gr.Column(scale=1):
                gr.Markdown("### 🔍 Evidence Board")
                context_display = gr.Textbox(label="Retrieved Context", lines=25, interactive=False)

        def user_turn(user_message, history):
            return "", history + [[user_message, None]]

        def bot_turn(history, top_k, temp):
            user_message = history[-1][0]
            
            # Run RAG
            sources_list, context_str = rag.retrieve(user_message, top_k)
            response = rag.generate(user_message, context_str, temp)
            
            # Update History
            history[-1][1] = response
            
            # Format sources
            sources_text = "\n---\n".join(sources_list) if sources_list else "No relevant context found."
            
            return history, sources_text

        msg.submit(user_turn, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot_turn, [chatbot, k_slider, temp_slider], [chatbot, context_display]
        )
        submit_btn.click(user_turn, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot_turn, [chatbot, k_slider, temp_slider], [chatbot, context_display]
        )
        clear_btn.click(lambda: None, None, chatbot, queue=False)

    manual_demo.launch(share=True, debug=True)

if __name__ == "__main__":
    launch_interface()
```

## Retrieval Augmented Generation in Action

<!-- #region editable=true slideshow={"slide_type": ""} -->
While the RAG pipeline in the previous section is sufficient to demonstrate the principle, we were keen to push for even greater accuracy, interactivity, and above all, maintain a close emphasis on the actual sources to make verifiability as easy as possible. The clip below demonstrates the finished user interface, which we built as a Streamlit app (this is not currently available to the public). Notice how the researcher can switch between Presidents, constrain their answers by time period for more specific responses, and ask broad and open questions. Crucially, it is possible to move seamlessly from the generated output (which represents verbatim the underlying retrieved sources with about 98% fidelity), to the retrieved chunks, and finally the original speeches themselves as hosted on the American Presidency Project website. In this 'production' version, we used [Pinecone](https://www.pinecone.io/) for vector storage, [Voyage AI](https://www.voyageai.com/) for embedding and reranking models, [Anthropic's Claude Sonnet 4 model](https://www.anthropic.com/news/claude-4), and [Streamlit](https://streamlit.io/) for the user interface. At one stage, this tool also included automatic speech-to-text, using cloned versions of the president's voices, courtesy of [Elevenlabs](https://elevenlabs.io/). 
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""} tags=["video-app_demo-*"]
from IPython.display import VimeoVideo, display
display(VimeoVideo('1163638412','100%','347'))
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
As you can see, the RAG app above remains focused on the real historical sources. While the LLM is a useful addition to facilitate something approaching a conversation with the archive, it does introduce an interpretive layer between the historian and their sources. It makes selections, omits some quotes, and chooses others. The reasons for these choices are inscrutable, which is potentially problematic. One could easily imagine this app retaining much of its utility without the LLM in the 'generation' phase. The embedding and result reranking models, and the semantic search in the 'retrieval' phase, contribute to the core value from the academic historian's perspective. Nevertheless, opportunities exist to harness LLMs to make history more interactive and encourage public participation. We aimed to highlight this potential by hosting an 'AI Presidential Debate', which received a large and diverse turnout. A clip from this debate can be viewed below.
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""} tags=["video-debate_demo-*"]
from IPython.display import VimeoVideo, display
display(VimeoVideo('1163653933','100%','347'))
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
## Finetuning for Stylistic Alignment
<!-- #endregion -->

<!-- #region citation-manager={"citations": {"pmp6h": [{"id": "22606145/8287VKJ3", "source": "zotero"}], "si6y5": [{"id": "22606145/5NARYZK9", "source": "zotero"}]}} editable=true slideshow={"slide_type": ""} -->
While RAG primarily addresses the retrieval of specific historical facts, it does not inherently address the voice of the historical figure. There is substantial evidence to suggest that declarative knowledge can be transferred into Large Language Models via context injection and fine-tuning <cite id="si6y5"><a href="#zotero%7C22606145%2F5NARYZK9">(Ovadia et al., 2024)</a></cite>, <cite id="pmp6h"><a href="#zotero%7C22606145%2F8287VKJ3">(Mecklenburg et al., 2024)</a></cite>. However, there has been strikingly little research conducted into the transfer of speaker style onto an LLM in the absence of prompt engineering.
<!-- #endregion -->

<!-- #region citation-manager={"citations": {"uhnml": [{"id": "22606145/PQQBQBVS", "source": "zotero"}]}} editable=true slideshow={"slide_type": ""} -->
Famously, Frederick Mosteller and David Wallace conducted a landmark study in the 1960s that used statistical analysis to determine the disputed authorships of several of the Federalist Papers <cite id="uhnml"><a href="#zotero%7C22606145%2FPQQBQBVS">(Mosteller &#38; Wallace, 1963)</a></cite>. Their analysis made heavy use of commonly occurring words in the English language, so-called "stop words" or "function words" (e.g., "the," "and," "to"). Unlike content words, which depend heavily on the specific topic being discussed, function words are used unconsciously and at stable rates by specific speakers, acting as a stylistic fingerprint.
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} -->
In this section, we explore whether supervised fine-tuning (SFT) can transfer this stylistic fingerprint onto an LLM. Our primary measure of success will not be factual accuracy, but the distribution of function words used across presidential answers given to questions from the press during press conferences. It is important to note here that we are not optimising for the distribution of function words - the optimisation remains to minimise the cross-entropy loss of the output token distributions with the true token distributions. Rather, this alignment happens naturally from this process.
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} -->
## Finetuning Data Preparation
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} -->
For this experiment, we utilised the corpus of President Ronald Reagan’s press conferences. We randomly sampled 800 question and answer pairs, splitting them into training ($N=400$), validation ($N=200$), and testing ($N=200$) sets. The data is stored in `jsonl` format, in accordance with the specifications for [OpenAI](https://openai.com/)'s fine-tuning API. Each entry consists of a conversational turn formatted as a dictionary with `role` and `content` keys. The role points to one of three values: `system`, `user`, or `assistant`. To ensure the model remains grounded in the persona, we apply a consistent system message:

    
    You are President of the United States of America, Ronald Reagan. You are currently in a press conference. You will be asked a question by a member of the press. You will answer the question. Do not include any non-verbal information in your response.
   
The user message contains the query from the press member, and the assistant message contains the President's historical response. The goal of the language model is to reproduce the assistant message, token by token, minimising the cross-entropy loss between the predicted token distribution and the true historical tokens.
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""} tags=["hermeneutics"]
import asyncio
import base64
import json
import os
import string
from typing import List

import dotenv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from openai import OpenAI
from openai import AsyncOpenAI
from openai.types.fine_tuning import SupervisedMethod, SupervisedHyperparameters

from scipy import stats

from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from rich.pretty import pprint

dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI()
```

```python editable=true slideshow={"slide_type": ""} tags=["hermeneutics"]
with open("data/Reagan/all_qa_pairs.json", "r") as f:
    prompt_args_list = json.load(f)


random_indices = np.random.choice(len(prompt_args_list), size=800, replace=False)
sampled_pairs = [prompt_args_list[i] for i in random_indices]

# test train split 400:200:200 from sampled_pairs
train_pairs = sampled_pairs[:400]
validation_pairs = sampled_pairs[400:600]
test_pairs = sampled_pairs[600:]
```

```python editable=true slideshow={"slide_type": ""} tags=["hermeneutics"]
system_prompt = "You are President of the United States of America, Ronald Reagan. You are currently in a press conference. You will be asked a question by a member of the press. You will answer the question. Do not include any non-verbal information in your response."

def format_pairs_to_jsonl(pairs: List[dict], system_prompt: str, save_target: str = None, split: str = None) -> List[dict]:
    jsonl_data = []
    for pair in pairs:
        jsonl_data.append({"messages" : [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": pair["question"]},
            {"role": "assistant", "content": pair["answer"]}
        ]})
    if save_target and split:
        with open(f"data/Reagan/{split}_data.jsonl", "w", encoding="utf-8") as f:
            for item in jsonl_data:
                f.write(json.dumps(item) + "\n")
    return jsonl_data

train_jsonl = format_pairs_to_jsonl(train_pairs, system_prompt, save_target=True, split="train")
validation_jsonl = format_pairs_to_jsonl(validation_pairs, system_prompt, save_target=True, split="validation")
test_jsonl = format_pairs_to_jsonl(test_pairs, system_prompt, save_target=True, split="test")


```

```python editable=true slideshow={"slide_type": ""} tags=["hermeneutics"]
client.files.create(
  file=open("data/Reagan/train_data.jsonl", "rb"),
  purpose="fine-tune",
  expires_after={
    "anchor": "created_at",
    "seconds": 2592000
  }
)

client.files.create(
  file=open("data/Reagan/validation_data.jsonl", "rb"),
  purpose="fine-tune",
  expires_after={
    "anchor": "created_at",
    "seconds": 2592000
  }
)

job = client.fine_tuning.jobs.create(
  training_file="file-93GnaJ7wX9JoXYqW9zNShr",
  validation_file="file-9NmyR5PQTYiGnkp8ycnMKQ",
  model="gpt-4.1-mini-2025-04-14",
  method={
    "type": "supervised",
    "supervised": SupervisedMethod(
      hyperparameters=SupervisedHyperparameters(
        n_epochs=3
      )
    )
  },
  suffix="reagan-001"
)
```

```python editable=true slideshow={"slide_type": ""} tags=["hermeneutics"]
results = client.fine_tuning.jobs.retrieve(job.id)
pprint(results)
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
## Training Dynamics
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} -->
We utilised OpenAI's `gpt-4.1-mini-2025-04-14` as our base model. The training job was configured for 3 epochs using the standard supervised method. All other hyperparameters were fixed to the default values. Monitoring the training loss against the validation loss is critical in stylometric fine-tuning. Unlike factual fine-tuning, where overfitting might manifest as the memorisation of specific facts, overfitting in style transfer can result in the model producing caricatures of the speaker's syntax or repetitive verbal tics.

We retrieve the training metrics from the OpenAI API to visualise the convergence of the model. The following plot illustrates the training trajectory. We observe the training loss (blue) and validation loss (red) over the course of the training steps.
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""} tags=["hermeneutics"]
content = client.files.content(results.result_files[0])
base64.b64decode(content.text.encode('utf-8'))
with open('result.csv', 'wb') as f:
    f.write(base64.b64decode(content.text.encode('utf-8')))

df = pd.read_csv('result.csv')
df.head()
```

```python editable=true slideshow={"slide_type": ""} tags=["hermeneutics"]
# get validation loss without the nans
df_valid = df[df['valid_loss'].notna()]
df_valid.head(12)
```

```python editable=true slideshow={"slide_type": ""} tags=["hermeneutics"]
plt.figure(figsize=(6, 4), dpi=100)
plt.plot(df['step'], df['train_loss'], c='b', label='train loss', lw=0.5)
plt.plot(df_valid['step'], df_valid['valid_loss'], c='r', label='validation loss', lw=1)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.xlim(0, df['step'].max())
plt.show()
```

```python editable=true slideshow={"slide_type": ""} tags=["figure-plot_training_loss-*"]
from IPython.display import Image, display

display(Image("./media/plot_training_loss.png", width=750))
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
## Stylometric Evaluation
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} -->
To determine if the fine-tuning process successfully transferred President Reagan's \"voice,\" we rely on the function word frequency analysis inspired by Mosteller and Wallace. We generated responses to the held-out test set ($N=200$) using four model states: the base model with no fine-tuning, and the fine-tuned model at three checkpoints (one after each epoch). We then calculate the \"Function Word Ratio\" for every response. This is defined as,
    $$
    R_f = \\frac{C_{func}}{C_{total}}
    $$
    where $C_{func}$ is the count of stop words (derived from the NLTK English stop word list) and $C_{total}$ is the total count of non-punctuation tokens in the response.
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""} tags=["hermeneutics"]
checkpoints = []

for model in client.models.list():
    if 'reagan' in model.id:
        checkpoints.append(model.id)
        print(model.id)
```

```python editable=true slideshow={"slide_type": ""} tags=["hermeneutics"]
response = client.responses.create(
    model="ft:gpt-4.1-mini-2025-04-14:accelerate-science:reagan-001:CgVcylUi:ckpt-step-400",
    input=sample,
)

print(response.output_text)
```

```python editable=true slideshow={"slide_type": ""} tags=["hermeneutics"]
test_data = []
test_answers = []
with open('data/Reagan/test_data.jsonl', 'r') as f:
    for line in f:
        test_data.append(json.loads(line)['messages'][0:2])
        test_answers.append(json.loads(line)['messages'][-1]['content'])
```

```python editable=true slideshow={"slide_type": ""} tags=["hermeneutics"]
async def get_completion_async(message, model, client, semaphore):
    async with semaphore:
        completion = await client.chat.completions.create(
            model=model,
            messages=message
        )
        return completion.choices[0].message.content

async def get_answers_async(messages, model='gpt-4.1-mini-2025-04-14', max_concurrent_requests=5):
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(max_concurrent_requests)  # limit to 5 concurrent requests
    tasks = [get_completion_async(message, model, client, semaphore) for message in messages]
    answers = await async_tqdm.gather(*tasks, desc="Processing inputs")
    return answers
```

```python editable=true slideshow={"slide_type": ""} tags=["hermeneutics"]
test_answers_base = await get_answers_async(test_data, model='gpt-4.1-mini-2025-04-14', max_concurrent_requests=32)
test_answers_tune_epoch_1 = await get_answers_async(test_data, model='ft:gpt-4.1-mini-2025-04-14:accelerate-science:reagan-001:CgVcylUi:ckpt-step-400', max_concurrent_requests=32)
test_answers_tune_epoch_2 = await get_answers_async(test_data, model='ft:gpt-4.1-mini-2025-04-14:accelerate-science:reagan-001:CgVcyHFz:ckpt-step-800', max_concurrent_requests=32)
test_answers_tune_epoch_3 = await get_answers_async(test_data, model='ft:gpt-4.1-mini-2025-04-14:accelerate-science:reagan-001:CgVcz8NP', max_concurrent_requests=32)
```

```python editable=true slideshow={"slide_type": ""} tags=["hermeneutics"]
def _calculate_function_word_frequency(texts: list) -> list:
    """_summary_

    Args:
        words (list): _description_

    Returns:
        float: _description_
    """
    stop_words = set(stopwords.words('english'))
    function_word_ratios = []
    pbar = tqdm(
        texts,
        desc="🔍 Analyzing texts",
        unit="text",
        ncols=100,  # Width of progress bar
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        colour="green",
    )

    for text in pbar:
        words = word_tokenize(text.lower())
        words_no_punct = [word for word in words if word not in string.punctuation]

        function_word_count = sum(1 for word in words_no_punct if word in stop_words)

        ratio = function_word_count / len(words_no_punct) if words_no_punct else 0

        function_word_ratios.append(ratio)

    return function_word_ratios
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
## Results
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} -->
To visualise the shift in style, we plot the Kernel Density Estimation (KDE) of the function word ratios for all model states against the ground truth (the actual answers given by Reagan). The hypothesis is that the distribution of the Base Model will be significantly different from the True Responses, reflecting a generic "AI assistant" style. As fine-tuning progresses through Epochs 1, 2, and 3, we expect the distribution curves to shift and reshape to align more closely with the Green curve (True Responses). The KDE plot demonstrates the efficacy of the stylistic transfer. The overlap between the fine-tuned models and the ground truth indicates that the model has internalised the syntactic rhythm and function word usage of the historical figure, distinct from the generic phrasing of the base model.
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""} tags=["hermeneutics"]
def get_kde(function_word_freq, color, label):
    data = np.array(function_word_freq)
    kde = stats.gaussian_kde(data)
    x_range = np.linspace(0, 1, 1000)
    kde_values = kde(x_range)
    plt.plot(
        x_range,
        kde_values,
        color=color,
        linewidth=1,
        label=label,
    )

get_kde(function_word_freq_base, color='blue', label='Base Model')
get_kde(function_word_freq_tune, color='orange', label='Fine-tuned Model Epoch 1')
get_kde(function_word_freq_tune_2, color='red', label='Fine-tuned Model Epoch 2')
get_kde(function_word_freq_tune_3, color='purple', label='Fine-tuned Model Epoch 3')
get_kde(function_word_freq_true, color='green', label='True Responses')

plt.xlabel('Function Word Frequency')
plt.ylabel('Number of Responses')
plt.title('Function Word Frequency Distribution')
plt.legend()
plt.ylim(0, None)
plt.grid()
plt.show()
```

```python editable=true slideshow={"slide_type": ""} tags=["figure-function_word_frequency-*"]
from IPython.display import Image, display

display(Image("./media/function_word_frequency.png", width=750))
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
## Discussion
<!-- #endregion -->

<!-- #region citation-manager={"citations": {"ko2xr": [{"id": "22606145/Z75YKV7A", "source": "zotero"}]}} editable=true slideshow={"slide_type": ""} -->
The juxtaposition of RAG and fine-tuning within this study is more than just a technical comparison; it presents two distinct epistemological frameworks for navigating the "age of abundance" <cite id="ko2xr"><a href="#zotero%7C22606145%2FZ75YKV7A">(Milligan, 2019)</a></cite>. Where traditional digital history often relies on keyword frequency or topic modelling to aggregate data, the application of Large Language Models introduces a semantic layer to historical inquiry. As demonstrated by our RAG system, the utility of the LLM lies not in its generative creativity but in its ability to function as a semantic search engine. By vectorising millions of words of presidential speech, the model allows the historian to query the archive conceptually without relying on exact keyword matches, and the retrieval mechanism anchors the model's output strictly to the primary source text, helping mitigate the risks of hallucination inherent in these models.
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} -->
Conversely, the fine-tuning experiments performed on the Q&A corpus reveal the potential of LLMs to act as tools for simulating historical voices. By training a model to simulate the lexicon and rhetorical structures of a given president's speeches, we move from retrieving information to modelling discourse. The decline in training loss observed during this process serves as a quantitative proxy for the model's internalisation of a specific historical voice. If the RAG approach acts as a microscope for locating specific evidence within a vast archive, then fine-tuning functions as a simulation, allowing historians to generate synthetic texts that test our understanding of how historical actors constructed their arguments and used language. Respectively, these techniques cover two key areas of historical research activity: retrieval of text and discourse analysis. 
<!-- #endregion -->

Ultimately, these methodologies are complementary rather than mutually exclusive. The RAG architecture, demonstrated in our debate interface, addresses the crisis of abundance by curating relevance from noise, while preserving the provenance of the data, which is so crucial for scholarly verification. Meanwhile, fine-tuning offers a hermeneutic tool, enabling the historian to "read" the implicit biases and linguistic habits of a corpus by interacting with a model successfully trained upon it. As these tools evolve, the role of the historian may shift from solely being a reader of texts to becoming an architect of systems capable of reading at scale, requiring a critical understanding of both the archival source material and the algorithmic systems that process it.


## Conclusion


This study has investigated the potential of Large Language Models (LLMs) to transform historical research, contrasting two distinct methodologies: Retrieval Augmented Generation (RAG) and fine-tuning. By applying these techniques to corpora of presidential speeches, we have demonstrated that LLMs can serve as both powerful navigational aids for exploring vast archives and as hermeneutic tools for modelling historical discourse. Our findings suggest that these two approaches are not mutually exclusive but rather offer complementary epistemological benefits. RAG prioritises accuracy and provenance, making it suitable for evidence gathering and verification, while fine-tuning facilitates a deeper, more empathetic engagement with the source material by simulating its linguistic structures. As the "Training Loss" plot illustrates, the model's ability to internalise the statistical properties of presidential language improves with training, validating the use of fine-tuning for capturing the nuances of historical texts.


The integration of LLMs into the historian's toolkit necessitates a shift in disciplinary practice. Historians must become adept not only at close reading but also at "distant reading" through algorithmic means, critically evaluating the outputs of these models against traditional historical methods. Future research might focus on refining these techniques, exploring their applicability to non-English and multimodal sources, and to historical language less well-suited to embedding models trained on contemporary English. Establishing robust ethical guidelines for the use of such tools in academic scholarship will also be key.

<!-- #region citation-manager={"citations": {"28q7o": [{"id": "22606145/658Q2DB6", "source": "zotero"}], "ogzpn": [{"id": "22606145/UAPTYNS9", "source": "zotero"}], "udzhg": [{"id": "22606145/XG5LV6XR", "source": "zotero"}], "w9lrs": [{"id": "22606145/WUKNYC5F", "source": "zotero"}]}} -->
Ultimately, the stakes involved in this kind of research can be clearly seen in the debate within the digital history community about passivity versus proactivity in adopting new technology. Putnam astutely observes that “shifting the outer bound of the possible matters less than shifting the center of the easy,” hence why mass adoption of some digital technologies such as word search by historians has felt intuitive rather than revolutionary, and consequently hasn’t been sufficiently critiqued or theorised <cite id="ogzpn"><a href="#zotero%7C22606145%2FUAPTYNS9">(Putnam, 2016)</a></cite>. In 2003, Rosenzweig envisioned a series of bold changes occurring in undergraduate and graduate programs in history, as historians recognised and came to terms with the digital revolution and upskilled accordingly <cite id="w9lrs"><a href="#zotero%7C22606145%2FWUKNYC5F">(Rosenzweig, 2003)</a></cite>. Over two decades later, this widespread change has failed to materialise. This is doubly to our disciplinary detriment since historians have agency as users of technology and rather than being passive adopters can influence “the direction of technological change" <cite id="28q7o"><a href="#zotero%7C22606145%2F658Q2DB6">(Fridlund et al., 2020)</a></cite>. Cohen reiterates this crucial point, arguing that despite our natural preoccupation with the past “it would be a shame if we ceded the possibility of shaping the future to others” <cite id="udzhg"><a href="#zotero%7C22606145%2FXG5LV6XR">(“Interchange: The Promise of Digital History,” 2008)</a></cite>. Yet more pressingly, all of these calls for action over reaction stem from literature focused on the internet revolution, which began over three decades ago. We are all now, whether we choose to acknowledge it or not, historians in the age of AI. This project has offered two experimental avenues to constructively engage with AI in historical research, in a way that can harness its many benefits while remaining cognisant of its ethical and epistemological limitations. 
<!-- #endregion -->


