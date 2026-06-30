from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict

# User Schemas
class UserBase(BaseModel):
    name: str
    email: str

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: int
    role: str

    model_config = ConfigDict(from_attributes=True)

class UserLoginRequest(BaseModel):
    email: str
    password: str

class UserAuthResponse(BaseModel):
    success: bool
    message: str
    user: Optional[UserResponse] = None

# Image Schemas
class ImageBase(BaseModel):
    filename: str
    original_path: str

class ImageCreate(ImageBase):
    pass

class ImageResponse(BaseModel):
    id: int
    filename: str
    upload_time: datetime
    cloud_percentage: Optional[float] = None
    original_path: str
    mask_path: Optional[str] = None
    output_path: Optional[str] = None
    reconstruction_model: Optional[str] = None
    reconstruction_engine: Optional[str] = None
    reconstruction_badge: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class ReconstructionEngineInfo(BaseModel):
    engine_name: str
    engine_badge: str

class DetectCloudRequest(BaseModel):
    image_id: int

class ReconstructRequest(BaseModel):
    image_id: int

# Database Configuration Schemas
class DatabaseConfigureRequest(BaseModel):
    database_url: str

class DatabaseConfigureResponse(BaseModel):
    success: bool
    message: str

class DatabaseStatusResponse(BaseModel):
    database_type: str
    database_url: str
    is_connected: bool
    total_records: int
    total_users: int
    file_size_bytes: Optional[int] = None
    error_message: Optional[str] = None

