from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from embeddings import COLLECTION_NAME, VECTOR_SIZE, load_model, embed
import csv

QDRANT_URL = "http://localhost:6333"
RFP_CSV_FILE = "RFP_Questions.csv"
model = load_model()

client = QdrantClient(url=QDRANT_URL)

if(client.collection_exists(collection_name=COLLECTION_NAME)):
    client.delete_collection(collection_name=COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
)

csv_list = []
with open(RFP_CSV_FILE, newline = '') as rfp_csv:
    rfp_reader = csv.reader(rfp_csv, delimiter=',')
    for row in rfp_reader:
        csv_list.append({"id": row[0], "question": row[1], "answer": row[2]})

points = []
for r in csv_list:
    points.append(PointStruct(
        id=int(r["id"]), 
        vector=embed(model, r["question"]),
        payload=r
    ))

operation_info = client.upsert(
    collection_name=COLLECTION_NAME,
    wait=True,
    points=points,
)

print(operation_info)