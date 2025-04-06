
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from typing import List
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

df = pd.read_csv("shl_catalog.csv")
embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
vectorstore = FAISS.from_texts(df["embedding_text"].tolist(), embedding_model, metadatas=df.to_dict("records"))

@app.post("/recommend")
async def recommend(req: QueryRequest):
    results = vectorstore.similarity_search_with_score(req.query, k=10)
    top_results = []
    for doc, score in results:
        metadata = doc.metadata
        top_results.append({
            "Assessment Name": metadata.get("Assessment Name", ""),
            "URL": metadata.get("URL", ""),
            "Remote Testing Support": metadata.get("Remote", ""),
            "Adaptive/IRT Support": metadata.get("Adaptive", ""),
            "Duration": metadata.get("Duration", ""),
            "Test Type": metadata.get("Type", ""),
            "Score": score
        })
    return {"recommendations": top_results}
