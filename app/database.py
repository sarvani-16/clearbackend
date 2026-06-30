import datetime
import logging
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import settings

logger = logging.getLogger("cloudclear_db")
logging.basicConfig(level=logging.INFO)

Base = declarative_base()

# Attempt to initialize Database Engine
try:
    logger.info(f"Attempting to connect to PostgreSQL database: {settings.DATABASE_URL.split('@')[-1]}")
    # Set a short timeout (e.g., 3 seconds) for initial connection check
    engine = create_engine(
        settings.DATABASE_URL, 
        pool_pre_ping=True,
        connect_args={"connect_timeout": 3}
    )
    # Test the connection
    with engine.connect() as conn:
        logger.info("Successfully connected to PostgreSQL database!")
except Exception as e:
    logger.warning(f"Failed to connect to PostgreSQL: {e}. Falling back to SQLite database.")
    engine = create_engine(
        settings.SQLITE_FALLBACK_URL, 
        connect_args={"check_same_thread": False}
    )
    logger.info(f"SQLite database initialized at: {settings.SQLITE_FALLBACK_URL}")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def test_db_connection(url: str) -> tuple[bool, str]:
    """Test if a database connection URL is valid and reachable."""
    try:
        if url.startswith("sqlite"):
            temp_engine = create_engine(url, connect_args={"check_same_thread": False})
        else:
            temp_engine = create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 3})
            
        with temp_engine.connect() as conn:
            pass
        return True, "Connection successful"
    except Exception as e:
        return False, str(e)

def reconnect_database(url: str) -> tuple[bool, str]:
    """
    Dynamically update the active SQLAlchemy engine and SessionLocal.
    Also creates tables if they do not exist.
    """
    global engine, SessionLocal
    success, msg = test_db_connection(url)
    if not success:
        return False, f"Failed to connect to database URL: {msg}"
        
    try:
        if url.startswith("sqlite"):
            new_engine = create_engine(url, connect_args={"check_same_thread": False})
        else:
            new_engine = create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 3})
            
        engine = new_engine
        SessionLocal.configure(bind=engine)
        
        # Ensure tables are created on the new database connection
        Base.metadata.create_all(bind=engine)
        logger.info(f"Database successfully reconnected to: {url.split('@')[-1] if '@' in url else url}")
        return True, "Database successfully reconnected and initialized"
    except Exception as e:
        logger.error(f"Error swapping database engine: {e}")
        return False, f"Error swapping database engine: {str(e)}"

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=True)
    role = Column(String, default="user")

class Image(Base):
    __tablename__ = "images"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    filename = Column(String, nullable=False)
    upload_time = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    cloud_percentage = Column(Float, nullable=True)
    original_path = Column(String, nullable=False)
    mask_path = Column(String, nullable=True)
    output_path = Column(String, nullable=True)
    reconstruction_model = Column(String, nullable=True)
    user_id = Column(Integer, nullable=True)

def init_db():
    logger.info("Creating database tables if they do not exist...")
    Base.metadata.create_all(bind=engine)
    
    # Check and run database migrations dynamically
    db = SessionLocal()
    try:
        from sqlalchemy import inspect, text
        inspector = inspect(engine)
        columns = [c['name'] for c in inspector.get_columns('users')]
        if 'password' not in columns:
            logger.info("Migrating database: adding 'password' column to users table...")
            db.execute(text("ALTER TABLE users ADD COLUMN password VARCHAR"))
            db.commit()
        if 'role' not in columns:
            logger.info("Migrating database: adding 'role' column to users table...")
            db.execute(text("ALTER TABLE users ADD COLUMN role VARCHAR DEFAULT 'user'"))
            db.commit()
            
        img_columns = [c['name'] for c in inspector.get_columns('images')]
        if 'user_id' not in img_columns:
            logger.info("Migrating database: adding 'user_id' column to images table...")
            db.execute(text("ALTER TABLE images ADD COLUMN user_id INTEGER"))
            db.commit()
    except Exception as ex:
        logger.error(f"Error during database migrations: {ex}")
        db.rollback()

    # Check if a default test user and predefined admins exist, otherwise create them
    try:
        import hashlib
        
        default_user = db.query(User).filter(User.email == "demo@cloudclear.ai").first()
        if not default_user:
            logger.info("Creating demo user...")
            pw_hash = hashlib.sha256("demo123".encode()).hexdigest()
            demo = User(name="Demo User", email="demo@cloudclear.ai", password=pw_hash, role="user")
            db.add(demo)
            db.commit()
            
        admin1 = db.query(User).filter(User.email == "admin1@cloudclear.ai").first()
        if not admin1:
            logger.info("Creating predefined admin1...")
            pw_hash = hashlib.sha256("admin123".encode()).hexdigest()
            admin1 = User(name="Admin One", email="admin1@cloudclear.ai", password=pw_hash, role="admin")
            db.add(admin1)
            db.commit()

        admin2 = db.query(User).filter(User.email == "admin2@cloudclear.ai").first()
        if not admin2:
            logger.info("Creating predefined admin2...")
            pw_hash = hashlib.sha256("admin456".encode()).hexdigest()
            admin2 = User(name="Admin Two", email="admin2@cloudclear.ai", password=pw_hash, role="admin")
            db.add(admin2)
            db.commit()
            
        ai_agent = db.query(User).filter(User.email == "agent@cloudclear.ai").first()
        if not ai_agent:
            logger.info("Creating AI Agent...")
            pw_hash = hashlib.sha256("agent123".encode()).hexdigest()
            ai_agent = User(name="AI Agent", email="agent@cloudclear.ai", password=pw_hash, role="agent")
            db.add(ai_agent)
            db.commit()
            
    except Exception as ex:
        logger.error(f"Error initializing default users/admins: {ex}")
        db.rollback()
    finally:
        db.close()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
