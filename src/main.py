import uvicorn



# Create database tables
create_table()

# Register routers
app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(basic_router, tags=["basic"])

# Root endpoint
@app.get("/")
def root():
    return {"message": "API is running", "status": "ok"}

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        port=8000,
        reload=True
    )

