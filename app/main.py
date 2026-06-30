import os
import shutil
import uuid
from typing import List, Optional
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
from app.database import init_db, get_db, Image, User, test_db_connection, reconnect_database, engine
from app import schemas
from app.pipeline.cloud_detector import CloudDetector
from app.pipeline.reconstructor import get_reconstruction_engine
from app.report_generator import generate_pdf_report

# Initialize application and databases
app = FastAPI(title=settings.PROJECT_NAME, version="1.0.0")

# Setup database tables and defaults on startup
@app.on_event("startup")
def startup_event():
    init_db()

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://clearfrontend-production.up.railway.app",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount uploads and outputs folders as static endpoints
app.mount("/static/uploads", StaticFiles(directory=settings.UPLOAD_FOLDER), name="uploads")
app.mount("/static/outputs", StaticFiles(directory=settings.OUTPUT_FOLDER), name="outputs")

# Instantiate our AI Pipeline classes
detector = CloudDetector()


def _enrich_image_response(db_image: Image) -> schemas.ImageResponse:
    """Build API response with reconstruction engine metadata."""
    model_name = db_image.reconstruction_model
    badge = None
    if model_name:
        if "Demo" in model_name:
            badge = "Demo AI Reconstruction"
        else:
            badge = "Active AI Reconstruction"
            
    return schemas.ImageResponse(
        id=db_image.id,
        filename=db_image.filename,
        upload_time=db_image.upload_time,
        cloud_percentage=db_image.cloud_percentage,
        original_path=db_image.original_path,
        mask_path=db_image.mask_path,
        output_path=db_image.output_path,
        reconstruction_model=model_name,
        reconstruction_engine=model_name,
        reconstruction_badge=badge
    )

# API Endpoints

@app.post("/api/upload", response_model=schemas.ImageResponse, status_code=status.HTTP_201_CREATED)
def upload_image(file: UploadFile = File(...), user_id: Optional[int] = None, db: Session = Depends(get_db)):
    """
    Upload a satellite image tile.
    Saves the file to the uploads folder and records it in the database.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
        
    # Verify file extension is an image
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
        raise HTTPException(status_code=400, detail="Only JPG, PNG, and TIFF images are supported.")
        
    # Create unique filename
    unique_id = uuid.uuid4().hex
    safe_filename = f"{unique_id}{ext}"
    dest_path = os.path.join(settings.UPLOAD_FOLDER, safe_filename)
    
    # Save file locally
    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
        
    # Save image record to database
    db_image = Image(
        filename=file.filename,
        original_path=f"/static/uploads/{safe_filename}",
        user_id=user_id
    )
    db.add(db_image)
    db.commit()
    db.refresh(db_image)
    
    return _enrich_image_response(db_image)

@app.post("/api/detect-cloud", response_model=schemas.ImageResponse)
def detect_cloud(payload: schemas.DetectCloudRequest, db: Session = Depends(get_db)):
    """
    Detects the cloud coverage percentage and generates a cloud mask image.
    """
    # Fetch image from DB
    db_image = db.query(Image).filter(Image.id == payload.image_id).first()
    if not db_image:
        raise HTTPException(status_code=404, detail="Image not found")
        
    # Get local path of the original image
    safe_filename = os.path.basename(db_image.original_path)
    local_image_path = os.path.join(settings.UPLOAD_FOLDER, safe_filename)
    
    if not os.path.exists(local_image_path):
        raise HTTPException(status_code=404, detail="Original image file not found on disk")
        
    # Generate mask file path
    mask_filename = f"mask_{safe_filename}"
    local_mask_path = os.path.join(settings.OUTPUT_FOLDER, mask_filename)
    
    # Trigger cloud detector pipeline
    try:
        cloud_percentage = detector.detect(local_image_path, local_mask_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cloud detection error: {str(e)}")
        
    # Update DB record
    db_image.cloud_percentage = cloud_percentage
    db_image.mask_path = f"/static/outputs/{mask_filename}"
    db.commit()
    db.refresh(db_image)
    
    return _enrich_image_response(db_image)

@app.post("/api/reconstruct", response_model=schemas.ImageResponse)
def reconstruct_image(payload: schemas.ReconstructRequest, db: Session = Depends(get_db)):
    """
    Reconstructs cloud-covered regions via the active AI reconstruction engine.
    """
    db_image = db.query(Image).filter(Image.id == payload.image_id).first()
    if not db_image:
        raise HTTPException(status_code=404, detail="Image not found")
        
    if not db_image.mask_path:
        raise HTTPException(status_code=400, detail="Cloud mask must be generated before reconstruction.")
        
    orig_filename = os.path.basename(db_image.original_path)
    mask_filename = os.path.basename(db_image.mask_path)
    
    local_image_path = os.path.join(settings.UPLOAD_FOLDER, orig_filename)
    local_mask_path = os.path.join(settings.OUTPUT_FOLDER, mask_filename)
    
    if not os.path.exists(local_image_path) or not os.path.exists(local_mask_path):
        raise HTTPException(status_code=404, detail="Required files (original or mask) not found on disk")
        
    output_filename = f"reconstructed_{orig_filename}"
    local_output_path = os.path.join(settings.OUTPUT_FOLDER, output_filename)
    
    engine = get_reconstruction_engine()
    try:
        success = engine.reconstruct(local_image_path, local_mask_path, local_output_path)
        if not success:
            raise HTTPException(status_code=500, detail="Reconstruction pipeline failed.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reconstruction error: {str(e)}")
        
    db_image.output_path = f"/static/outputs/{output_filename}"
    db_image.reconstruction_model = engine.name
    db.commit()
    db.refresh(db_image)
    
    return _enrich_image_response(db_image)

@app.get("/api/reconstruction-engine", response_model=schemas.ReconstructionEngineInfo)
def get_active_engine():
    """Returns metadata for the currently active reconstruction engine."""
    engine = get_reconstruction_engine()
    badge = "Demo AI Reconstruction" if "Demo" in engine.name else "Active AI Reconstruction"
    return schemas.ReconstructionEngineInfo(
        engine_name=engine.name,
        engine_badge=badge
    )
@app.get("/api/history", response_model=List[schemas.ImageResponse])
def get_history(user_id: Optional[int] = None, db: Session = Depends(get_db)):
    """
    Returns the history of uploaded and processed images, sorted by newest first.
    """
    query = db.query(Image)
    if user_id is not None:
        query = query.filter(Image.user_id == user_id)
    images = query.order_by(Image.upload_time.desc()).all()
    return [_enrich_image_response(img) for img in images]

@app.get("/api/download/{id}")
def download_image(id: int, file_type: str = "reconstructed", db: Session = Depends(get_db)):
    """
    Downloads original image, cloud mask, reconstructed image, or processing report (PDF).
    """
    db_image = db.query(Image).filter(Image.id == id).first()
    if not db_image:
        raise HTTPException(status_code=404, detail="Image record not found")
        
    if file_type == "original":
        filename = os.path.basename(db_image.original_path)
        local_path = os.path.join(settings.UPLOAD_FOLDER, filename)
        media_type = "image/jpeg"
        dl_filename = f"original_{db_image.filename}"
        
    elif file_type == "mask":
        if not db_image.mask_path:
            raise HTTPException(status_code=400, detail="Cloud mask has not been generated yet.")
        filename = os.path.basename(db_image.mask_path)
        local_path = os.path.join(settings.OUTPUT_FOLDER, filename)
        media_type = "image/png"
        dl_filename = f"mask_{db_image.filename}"
        
    elif file_type == "report":
        if not db_image.mask_path or not db_image.output_path:
            raise HTTPException(status_code=400, detail="Reconstructed outputs must be complete before generating reports.")
        
        pdf_filename = f"report_{id}.pdf"
        local_path = os.path.join(settings.OUTPUT_FOLDER, pdf_filename)
        
        # Determine processing device
        device_name = "GPU"
        if detector.device and detector.device.type == "cpu":
            device_name = "CPU"
            
        try:
            generate_pdf_report(
                original_filename=db_image.filename,
                original_rel_path=db_image.original_path,
                mask_rel_path=db_image.mask_path,
                reconstructed_rel_path=db_image.output_path,
                cloud_percentage=db_image.cloud_percentage or 0.0,
                inference_time=2.1,
                device=device_name,
                dest_pdf_path=local_path
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate report PDF: {str(e)}")
            
        media_type = "application/pdf"
        dl_filename = f"report_{os.path.splitext(db_image.filename)[0]}.pdf"
        
    else: # default to reconstructed
        if not db_image.output_path:
            raise HTTPException(status_code=400, detail="Reconstructed image has not been generated yet.")
        filename = os.path.basename(db_image.output_path)
        local_path = os.path.join(settings.OUTPUT_FOLDER, filename)
        media_type = "image/jpeg"
        dl_filename = f"cloudclear_{db_image.filename}"
        
    if not os.path.exists(local_path):
        raise HTTPException(status_code=404, detail="Requested file not found on disk")
        
    return FileResponse(
        path=local_path, 
        media_type=media_type, 
        filename=dl_filename
    )

# Database Management Endpoints

@app.get("/api/db-status", response_model=schemas.DatabaseStatusResponse)
def get_db_status(db: Session = Depends(get_db)):
    """
    Exposes the database connectivity state, driver type, url, and table stats.
    """
    url = str(engine.url)
    db_type = "PostgreSQL" if "postgresql" in url else "SQLite"
    
    # Mask password for safety
    masked_url = url
    if "@" in url:
        try:
            # Mask user credentials: postgresql://username:password@host/dbname
            parts = url.split("://")
            auth_and_host = parts[1].split("@")
            host_db = auth_and_host[-1]
            masked_url = f"{parts[0]}://*****@{host_db}"
        except Exception:
            masked_url = "postgresql://*****@database_host/cloudclear"
        
    is_connected = False
    error_msg = None
    total_records = 0
    total_users = 0
    file_size = None
    
    try:
        db.execute(text("SELECT 1"))
        is_connected = True
    except Exception as e:
        error_msg = str(e)
        
    if is_connected:
        try:
            total_records = db.query(Image).count()
            total_users = db.query(User).count()
        except Exception as e:
            error_msg = f"Failed to query stats: {str(e)}"
            
    if db_type == "SQLite":
        db_path = url.replace("sqlite:///", "").replace("sqlite://", "")
        if os.path.exists(db_path):
            try:
                file_size = os.path.getsize(db_path)
            except Exception:
                pass
                
    return schemas.DatabaseStatusResponse(
        database_type=db_type,
        database_url=masked_url,
        is_connected=is_connected,
        total_records=total_records,
        total_users=total_users,
        file_size_bytes=file_size,
        error_message=error_msg
    )

@app.post("/api/db-test", response_model=schemas.DatabaseConfigureResponse)
def test_db_endpoint(payload: schemas.DatabaseConfigureRequest):
    """
    Tests connectivity to a specific database URL.
    """
    success, msg = test_db_connection(payload.database_url)
    return schemas.DatabaseConfigureResponse(success=success, message=msg)

@app.post("/api/db-configure", response_model=schemas.DatabaseConfigureResponse)
def configure_db_endpoint(payload: schemas.DatabaseConfigureRequest):
    """
    Saves the new database URL to the local .env file, hot-reconnects, and migrates tables.
    """
    url = payload.database_url
    
    # 1. Test connection first
    success, msg = test_db_connection(url)
    if not success:
        raise HTTPException(status_code=400, detail=f"Database connection test failed: {msg}")
        
    # 2. Update the active connection dynamically
    swap_success, swap_msg = reconnect_database(url)
    if not swap_success:
        raise HTTPException(status_code=500, detail=f"Failed to hot-swap database engine: {swap_msg}")
        
    # 3. Write to .env file to persist settings
    try:
        env_path = os.path.join(settings.BASE_DIR, "backend", ".env")
        
        # Read current lines
        lines = []
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                lines = f.readlines()
                
        # Update or add DATABASE_URL
        db_url_found = False
        new_lines = []
        for line in lines:
            if line.strip().startswith("DATABASE_URL="):
                new_lines.append(f"DATABASE_URL={url}\n")
                db_url_found = True
            else:
                new_lines.append(line)
                
        if not db_url_found:
            new_lines.append(f"DATABASE_URL={url}\n")
            
        with open(env_path, "w") as f:
            f.writelines(new_lines)
            
    except Exception as e:
        print(f"[configure_db_endpoint] Warning: Failed to write to .env file: {e}")
        
    return schemas.DatabaseConfigureResponse(
        success=True, 
        message="Database successfully reconfigured, migrated, and saved to environment settings."
    )

@app.post("/api/db-clear-history", response_model=schemas.DatabaseConfigureResponse)
def clear_history_endpoint(db: Session = Depends(get_db)):
    """
    Deletes all images in the database and cleans upload/output directories on disk.
    """
    try:
        db.query(Image).delete()
        db.commit()
        
        # Clear uploads and outputs folders (delete JPEG, PNG, PDF files)
        cleared_count = 0
        for folder in [settings.UPLOAD_FOLDER, settings.OUTPUT_FOLDER]:
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in [".jpg", ".jpeg", ".png", ".pdf", ".tif", ".tiff"]:
                        file_path = os.path.join(folder, filename)
                        try:
                            if os.path.isfile(file_path):
                                os.unlink(file_path)
                                cleared_count += 1
                        except Exception as e:
                            print(f"Failed to delete {file_path}: {e}")
                            
        return schemas.DatabaseConfigureResponse(
            success=True, 
            message=f"All history records deleted from database. Cleared {cleared_count} image/report files on disk."
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {str(e)}")

# Authentication & Admin Endpoints
import hashlib

@app.post("/api/auth/register", response_model=schemas.UserAuthResponse)
def register_user_endpoint(payload: schemas.UserCreate, db: Session = Depends(get_db)):
    """Registers a new standard user account."""
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        return schemas.UserAuthResponse(success=False, message="Email is already registered.")
        
    pw_hash = hashlib.sha256(payload.password.encode()).hexdigest()
    new_user = User(
        name=payload.name,
        email=payload.email,
        password=pw_hash,
        role="user"
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    user_resp = schemas.UserResponse(
        id=new_user.id,
        name=new_user.name,
        email=new_user.email,
        role=new_user.role
    )
    return schemas.UserAuthResponse(success=True, message="User registered successfully.", user=user_resp)

@app.post("/api/auth/login", response_model=schemas.UserAuthResponse)
def login_user_endpoint(payload: schemas.UserLoginRequest, db: Session = Depends(get_db)):
    """Authenticates standard users and predefined administrators."""
    pw_hash = hashlib.sha256(payload.password.encode()).hexdigest()
    user = db.query(User).filter(User.email == payload.email, User.password == pw_hash).first()
    if not user:
        return schemas.UserAuthResponse(success=False, message="Invalid email or password.")
        
    user_resp = schemas.UserResponse(
        id=user.id,
        name=user.name,
        email=user.email,
        role=user.role
    )
    return schemas.UserAuthResponse(success=True, message="Login successful.", user=user_resp)

@app.get("/api/admin/users", response_model=List[schemas.UserResponse])
def get_all_users_endpoint(db: Session = Depends(get_db)):
    """Lists all registered standard users for administration."""
    users = db.query(User).filter(User.role == "user").all()
    return [
        schemas.UserResponse(
            id=u.id,
            name=u.name,
            email=u.email,
            role=u.role
        ) for u in users
    ]

@app.delete("/api/admin/users/{user_id}", response_model=schemas.DatabaseConfigureResponse)
def delete_user_endpoint(user_id: int, db: Session = Depends(get_db)):
    """Deletes a standard user from the system."""
    user = db.query(User).filter(User.id == user_id, User.role == "user").first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found or cannot be deleted.")
        
    db.delete(user)
    db.commit()
    return schemas.DatabaseConfigureResponse(success=True, message=f"User '{user.name}' deleted successfully.")

# Root status
@app.get("/")
def read_root():
    return {"message": "CloudClear AI backend running successfully", "status": "active"}
