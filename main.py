from fastapi import FastAPI, Query
from pydantic import BaseModel
import json
from sentence_transformers import SentenceTransformer
import torch

app = FastAPI()

# Load model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load courses
with open("courses.json", "r") as f:
    courses = json.load(f)

print("Loaded courses.json with", len(courses), "courses. Starting embedding...")

# Precompute embeddings
for course in courses:
    description = ''.join(courses[course]["learning-objectives"])
    embedded_objectives = model.encode(courses[course]["learning-objectives"], convert_to_tensor=True)
    courses[course]["embedded_objectives"] = embedded_objectives
    courses[course]["embedded_description"] = model.encode(description, convert_to_tensor=True)
    courses[course]["embedded_title"] = model.encode(courses[course]["title"], convert_to_tensor=True)

print("Courses and objectives have been embedded.")


# ---------------------- ROUTES ----------------------

@app.get("/v1/courses/{course_id}/similar")
def find_similar_courses(
    course_id: str,
    top_k: int = Query(10),
    mode: str = Query("dense", regex="^(dense|sparse|hybrid)$"),
    alpha: float = Query(0.5)
):
    """
    Search for courses based on a course ID.
    Compare the query course title and description embeddings with all other courses.
    Return the top_k most similar courses.
    """
    if course_id not in courses:
        return {"error": f"Course ID '{course_id}' not found."}

    query_title = courses[course_id]["embedded_title"]
    query_description = courses[course_id]["embedded_description"]

    similarities = {}
    for course in courses:
        title_embedding = courses[course]["embedded_title"]
        description_embedding = courses[course]["embedded_description"]

        title_title_similarity = torch.dot(query_title, title_embedding).item()
        title_description_similarity = torch.dot(query_title, description_embedding).item()
        description_title_similarity = torch.dot(query_description, title_embedding).item()
        description_description_similarity = torch.dot(query_description, description_embedding).item()

        similarities[course] = max(
            title_title_similarity,
            title_description_similarity,
            description_title_similarity,
            description_description_similarity
        )

    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    results = [
        {"course_id": c, "title": courses[c]["title"], "score": s}
        for c, s in sorted_similarities[:top_k]
    ]

    return {
        "query_course_id": course_id,
        "results": results,
        "mode": mode,
        "top_k": top_k,
    }


@app.get("/v1/search")
def search_courses(
    query: str = Query(...),
    top_k: int = Query(10),
    mode: str = Query("dense", regex="^(dense|sparse|hybrid)$"),
    alpha: float = Query(0.5)
):
    """
    Search for courses based on a free text query.
    Compare query embedding with titles and descriptions.
    Return top_k most similar courses.
    """
    query_embedding = model.encode(query.lower(), convert_to_tensor=True)
    similarities = {}

    for course in courses:
        title_embedding = courses[course]["embedded_title"]
        description_embedding = courses[course]["embedded_description"]
        title_similarity = torch.dot(query_embedding, title_embedding).item()
        description_similarity = torch.dot(query_embedding, description_embedding).item()
        similarities[course] = max(title_similarity, description_similarity)

    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    results = [
        {"course_id": c, "title": courses[c]["title"], "score": s}
        for c, s in sorted_similarities[:top_k]
    ]

    return {
        "query": query,
        "results": results,
        "mode": mode
    }


@app.get("/v1/objectives/search")
def search_objectives(
    query: str = Query(...),
    top_k: int = Query(10),
    mode: str = Query("dense", regex="^(dense|sparse|hybrid)$"),
    alpha: float = Query(0.5)
):
    """
    Search for individual course objectives based on a free text query.
    Return top_k most similar objectives and their courses.
    """
    query_embedding = model.encode(query.lower(), convert_to_tensor=True)
    similarities = {}

    for course in courses:
        for i, objective_embedding in enumerate(courses[course]["embedded_objectives"]):
            similarity = torch.dot(query_embedding, objective_embedding).item()
            objective = courses[course]["learning-objectives"][i]
            similarities[(course, objective)] = similarity

    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    results = []
    for (course, objective), sim in sorted_similarities[:top_k]:
        results.append({
            "course_id": course,
            "title": courses[course]["title"],
            "objective": objective,
            "score": sim
        })

    return {
        "query": query,
        "results": results
    }


@app.get("/v1/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
