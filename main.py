from fastapi import FastAPI
from pydantic import BaseModel
import json
from sentence_transformers import SentenceTransformer
app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/v1/courses/{course_id}/similar?top_k=10&mode=dense|sparse|hybrid&alpha=0.5")
def find_similar_courses(text: TextInput):
    lowered_text = text.text.lower()
    pass

@app.post("/v1/search?query=machine%20learning&top_k=10&mode=dense|sparse|hybrid&alpha=0.5")
def search_courses(query: str):
    lowered_query = query.lower()
    pass

@app.post("/v1/objectives/search?query=dimensionality%20reduction&top_k=10&mode=dense|sparse|hybrid&alpha=0.5")
def search_objectives(query: str):
    lowered_query = query.lower()
    pass

@app.post("/v1/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":

    QUERY = "machine learning"
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    with open("courses.json", "r") as f:
        courses = json.load(f)
    
    print("Loaded courses.json with", len(courses), "courses.")
    for course in courses:
        embedded_objectives = model.encode(courses[course]["learning-objectives"], convert_to_tensor=True)
        courses[course]["embedded_objectives"] = embedded_objectives
        courses[course]["embedded_title"] = model.encode(courses[course]["title"], convert_to_tensor=True)
    print(courses[course]["embedded_title"].shape)
    print(courses[course]["embedded_objectives"].shape)
    print("Courses and objectives have been embedded.")
    query_embedding = model.encode(QUERY, convert_to_tensor=True)
    similarities = {}
    for course in courses:
        title_embedding = courses[course]["embedded_title"]
        similarity = (query_embedding @ title_embedding.T).item()
        similarities[course] = similarity
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    print("Top courses for query:", QUERY)
    for course, sim in sorted_similarities[:10]:
        print(f"Course ID: {course}, Similarity: {sim:.4f}, Title: {courses[course]['title']}")