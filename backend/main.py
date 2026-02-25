from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import face
from app.database.face_db import init_db

app = FastAPI(
    title="University Attendance System API",
    description="Backend for face-recognition based attendance by CMPJ",
    version="1.0.0"
)

# Initialize database
init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(face.router)

@app.get("/")
async def root():
    return {"message": "Welcome to the University Attendance System API"}

if __name__ == "__main__":
    import uvicorn
    # reload=True is great for development!
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
