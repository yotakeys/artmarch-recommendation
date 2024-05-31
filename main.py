import gradio as gr
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os

class Karya:

    DATA_DIR = "data/"
    DATA_PATH = os.path.join(DATA_DIR, "data.csv")
    INDEX_PATH = os.path.join(DATA_DIR, "karya.index")
    MODEL_PATH = "model/"

    def __init__(self) -> None:
        self.model = SentenceTransformer(self.MODEL_PATH)
        self.index = faiss.read_index(self.INDEX_PATH)
        self.data = pd.read_csv(self.DATA_PATH)

    def reload(self):
        self.model = SentenceTransformer(self.MODEL_PATH)
        self.index = faiss.read_index(self.INDEX_PATH)
        self.data = pd.read_csv(self.DATA_PATH)

    def choice(self, option, uuid: str, key: str, top_k: int):
        if option == "add":
            return self.add_karya(uuid, key)
        elif option == "delete":
            return self.delete_karya(uuid)
        elif option == "search":
            return self.search_karya(key, top_k)

    
    def add_karya(self, uuid: str, key: str):

        # Add to index map
        len_data = self.data.shape[0]
        if len_data == 0:
            new_idx = 1
        else: 
            new_idx = self.data.iloc[-1]["idx"]+1
        encoded_data = self.model.encode([key])
        encoded_data = np.asarray(encoded_data.astype('float32'))
        self.index.add_with_ids(encoded_data, np.array([new_idx]))
        faiss.write_index(self.index, self.INDEX_PATH)

        # Add to csv
        new_record = pd.DataFrame({'idx': new_idx, 'uuid': [uuid], 'key': [key]})
        self.data = pd.concat([self.data, new_record], ignore_index=True)
        self.data.to_csv(self.DATA_PATH, index=False)

        return "Added"
    
    def delete_karya(self, uuid: str):
        # Delete from index map
        idx_df = self.data[self.data['uuid'] == uuid].index[0]
        idx = self.data[self.data['uuid'] == uuid]["idx"]
        self.index.remove_ids(np.asarray(idx))
        faiss.write_index(self.index, self.INDEX_PATH)
        
        # Delete from csv
        self.data = self.data.drop(idx_df)
        self.data.to_csv(self.DATA_PATH, index=False)

        return "Deleted"
    
    def search_karya(self, query: str, top_k: int):

        query_vector = self.model.encode([query])
        top_k_recommend = self.index.search(query_vector, top_k)
        result_id = top_k_recommend[1].tolist()[0]
        result_id = list(np.unique(result_id))
        results = []
        for result in result_id:
            results.append(self.data[self.data["idx"] == result]["uuid"].iloc[0])
        return results


karya = Karya()

app = gr.Interface(
    fn=karya.choice,
    inputs=["text", "text", "text", "number"],
    outputs=["text"],
)

app.launch()