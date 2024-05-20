import pandas as pd
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

datapath = "./App/initdata/"
# datapath = "./initdata/"
maxx = max

class VectorDB:
    def __init__(self,
                 bert_model=SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS'),
                 db_path=datapath+'vdb',
                 initial_data=datapath+'vector.xls',
                 similarity_metric='cosine',
                 chunk=50,
                 strong_threshold=0.2,
                 weak_threshold=0.7,
                 max_size=10000
                 ):

        np.set_printoptions(suppress=True, precision=2)

        self.bert_model = bert_model
        self.db_path = db_path
        # self.initial_data = initial_data
        self.similarity_metric = similarity_metric
        self.chunk = chunk
        self.strong_threshold = strong_threshold
        self.weak_threshold = weak_threshold
        self.max_size = max_size
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(
            name="coverletter",
            metadata={"hnsw:space": self.similarity_metric}
            )

        if self.collection.count() == 0:
            print("Initialize Vector DB")
            self.ids = []
            self.metadatas = []
            self.embeddings = []

            df = pd.read_excel(initial_data)

            for index, row in df.iterrows():
                prompt = row[0]
                suggestion = row[1]
                # print(index, prompt, suggestion)

                meta_data = {
                    'prompt' : prompt,
                    'suggestion' : suggestion
                }
                embedding = bert_model.encode(prompt, normalize_embeddings=True)

                self.ids.append(str(index))
                self.metadatas.append(meta_data)
                self.embeddings.append(embedding)

            self.current_index = len(self.embeddings)

            total_chuck = self.current_index // chunk + 1
            # print(total_chuck)
            embeddings_list = [i.tolist() for i in self.embeddings]

            for chunk_idx in range(total_chuck):
                start_idx = chunk_idx * chunk
                end_idx = (chunk_idx +1) * chunk

                chunk_embedding = embeddings_list[start_idx : end_idx]
                chunk_ids = self.ids[start_idx:end_idx]
                chunk_metadatas = self.metadatas[start_idx:end_idx]

                self.collection.add(embeddings=chunk_embedding, ids=chunk_ids, metadatas= chunk_metadatas)
        else:
            ids = self.collection.get({})["ids"]    # future improve
            self.current_index = maxx([int(x) for x in ids]) + 1
            print(F"VDB size: {self.collection.count()}")
            print(F"VDB index: {self.current_index}")

        self.buffer_position = 0
        self.ids = []
        self.metadatas = []
        self.embeddings = []

    def query(self, prompt, max_results=3):
        result = self.collection.query(
            query_embeddings=self.bert_model.encode(prompt, normalize_embeddings=True).tolist(),
            n_results=max_results
        )
        n_result = len(result['ids'][0])
        # print(n_result)
        print(result)
        result_list = []
        if result['distances'][0][0] < self.weak_threshold:
            result_list.append(result['metadatas'][0][0]['suggestion'])
            for i in range(1, n_result):
                if result['distances'][0][i] < self.strong_threshold:
                    match_flag = False
                    for j in range(len(result_list)):
                        # print(f"i{i}, j{j}, i_len{len(result['metadatas'][0])}, j_len{len(result_list)}")
                        if result_list[j] == result['metadatas'][0][i]['suggestion']:
                            match_flag = True
                    if match_flag == False:
                        result_list.append(result['metadatas'][0][i]['suggestion'])

        print(result_list)
        return result_list
        # return result['metadatas'][0][0]['suggestion']

    def flush(self):
        print("VDB flush")
        if self.buffer_position > 0:
            if self.collection.count() + len(self.ids) < self.max_size:
                embeddings_list = [i.tolist() for i in self.embeddings]
                self.collection.add(embeddings=embeddings_list, ids=self.ids, metadatas=self.metadatas)
            else:
                # future improve
                pass
            self.buffer_position = 0
            self.ids = []
            self.metadatas = []
            self.embeddings = []

    def add(self, prompt, suggestion, force=False):
        meta_data = {
            'prompt' : prompt,
            'suggestion' : suggestion
        }
        embedding = self.bert_model.encode(prompt, normalize_embeddings=True)

        self.ids.append(str(self.current_index))
        self.metadatas.append(meta_data)
        self.embeddings.append(embedding)
        self.current_index += 1
        self.buffer_position += 1

        if self.buffer_position == self.chunk or force:
            self.flush()
            # print("vector DB flushing")

    def vdb_prompt(self, generate_target, job_title, comp_name, comp_info):
        job_title = job_title.replace(comp_name, "")

        return (generate_target + ". " + job_title + ". " + comp_info).strip()

    def shutdown(self):
        self.flush()
        # close?

    def __del__(self):
        self.shutdown()

# vector_db = VectorDB()
# prompt = "꿈,열정,최고,도전,고객 프로그래머 강점"
# vectordb_suggestion = vector_db.query(prompt)
# print(vectordb_suggestion)
#
# prompt = "화려함, 워라벨, 한탕주의, 아이돌 포부"
# suggestion = "BTS보다 잘 나가는 아이돌이 되고 싶습니다."
# vector_db.add(prompt, suggestion, force=False)
#
# vectordb_suggestion = vector_db.query(prompt)
# print(vectordb_suggestion)
#
# vector_db.flush()
# vectordb_suggestion = vector_db.query(prompt)
# print(vectordb_suggestion)
#
# prompt = "이상한 값"
# vectordb_suggestion = vector_db.query(prompt)
# print(vectordb_suggestion)
#
# vector_db.shutdown()