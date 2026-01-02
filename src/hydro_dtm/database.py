"""
Database configuration and models for the Intelligent Hydro-DTM system.
"""

import os
from typing import Optional, Dict, Any, List
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, String, Integer, Float, DateTime, 
    Boolean, Text, JSON, ForeignKey, BigInteger, DECIMAL
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.dialects.postgresql import UUID
from geoalchemy2 import Geometry
import uuid

from .logging_config import get_db_logger
from .exceptions import DatabaseConnectionError

logger = get_db_logger()

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://hydro_user:hydro_pass@localhost:5432/hydro_dtm"
)

Base = declarative_base()


class Project(Base):
    """Projects table for organizing work."""
    __tablename__ = "projects"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    village_code = Column(String(50))  # SVAMITVA village code
    district = Column(String(100))
    state = Column(String(100))
    description = Column(Text)
    bounds = Column(Geometry('POLYGON', srid=4326))
    government_approval_status = Column(String(50), default='pending')
    svamitva_sync_status = Column(String(50), default='not_synced')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    point_clouds = relationship("PointCloudDataset", back_populates="project")
    waterlogging_predictions = relationship("WaterloggingPrediction", back_populates="project")
    drainage_networks = relationship("DrainageNetworkDB", back_populates="project")
    government_submissions = relationship("GovernmentSubmission", back_populates="project")
    quality_assessments = relationship("QualityAssessment", back_populates="project")


class AIModel(Base):
    """AI model versions and performance tracking."""
    __tablename__ = "ai_models"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_type = Column(String(100), nullable=False)  # 'ground_classifier', 'dtm_generator', etc.
    version = Column(String(50), nullable=False)
    accuracy_metrics = Column(JSON)
    training_data_info = Column(JSON)
    deployment_date = Column(DateTime)
    is_active = Column(Boolean, default=False)
    model_path = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)


class PointCloudDataset(Base):
    """Point cloud datasets."""
    __tablename__ = "point_clouds"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey('projects.id'), nullable=False)
    filename = Column(String(255), nullable=False)
    file_size = Column(BigInteger)
    point_count = Column(Integer)
    bounds = Column(Geometry('POLYGON', srid=4326))
    processing_status = Column(String(50), default='pending')
    metadata = Column(JSON)
    file_path = Column(String(500))
    coordinate_system = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    project = relationship("Project", back_populates="point_clouds")
    dtms = relationship("DTMDataset", back_populates="point_cloud")


class DTMDataset(Base):
    """Generated DTMs."""
    __tablename__ = "dtms"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    point_cloud_id = Column(UUID(as_uuid=True), ForeignKey('point_clouds.id'), nullable=False)
    resolution = Column(Float, nullable=False)
    bounds = Column(Geometry('POLYGON', srid=4326))
    quality_score = Column(Float)
    file_path = Column(String(500))
    generation_method = Column(String(100), default='ai_interpolation')
    uncertainty_file_path = Column(String(500))  # Path to uncertainty grid
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    point_cloud = relationship("PointCloudDataset", back_populates="dtms")
    hydro_analyses = relationship("HydroAnalysis", back_populates="dtm")


class HydroAnalysis(Base):
    """Hydrological analysis results."""
    __tablename__ = "hydro_analyses"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dtm_id = Column(UUID(as_uuid=True), ForeignKey('dtms.id'), nullable=False)
    analysis_type = Column(String(100))
    parameters = Column(JSON)
    results = Column(JSON)
    flow_direction_path = Column(String(500))
    flow_accumulation_path = Column(String(500))
    watersheds_path = Column(String(500))
    stream_network_path = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    dtm = relationship("DTMDataset", back_populates="hydro_analyses")


class WaterloggingPrediction(Base):
    """Waterlogging predictions and risk assessments."""
    __tablename__ = "waterlogging_predictions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey('projects.id'), nullable=False)
    prediction_date = Column(DateTime, default=datetime.utcnow)
    risk_grid_path = Column(String(500))  # Path to risk probability raster
    hotspot_polygons = Column(Geometry('MULTIPOLYGON', srid=4326))
    seasonal_variations = Column(JSON)
    model_confidence = Column(Float)
    validation_status = Column(String(50), default='pending')
    model_version = Column(String(50))
    
    # Relationships
    project = relationship("Project", back_populates="waterlogging_predictions")


class DrainageNetworkDB(Base):
    """Drainage network designs."""
    __tablename__ = "drainage_networks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey('projects.id'), nullable=False)
    design_version = Column(Integer, default=1)
    channels = Column(Geometry('MULTILINESTRING', srid=4326))
    nodes = Column(Geometry('MULTIPOINT', srid=4326))
    total_cost = Column(DECIMAL(15, 2))
    coverage_area = Column(Float)
    hydraulic_capacity = Column(JSON)
    government_approval_status = Column(String(50), default='pending')
    construction_priority = Column(Integer)
    design_parameters = Column(JSON)
    environmental_impact = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    project = relationship("Project", back_populates="drainage_networks")


class GovernmentSubmission(Base):
    """Government integration tracking."""
    __tablename__ = "government_submissions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey('projects.id'), nullable=False)
    submission_type = Column(String(100))  # 'svamitva', 'panchayat', 'revenue'
    submission_data = Column(JSON)
    submission_date = Column(DateTime, default=datetime.utcnow)
    response_data = Column(JSON)
    status = Column(String(50), default='pending')  # 'pending', 'approved', 'rejected'
    follow_up_required = Column(Boolean, default=False)
    
    # Relationships
    project = relationship("Project", back_populates="government_submissions")


class QualityAssessment(Base):
    """Quality assurance and validation results."""
    __tablename__ = "quality_assessments"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey('projects.id'), nullable=False)
    assessment_type = Column(String(100))
    quality_metrics = Column(JSON)
    validation_points = Column(Geometry('MULTIPOINT', srid=4326))
    accuracy_statistics = Column(JSON)
    issues_identified = Column(JSON)
    remediation_suggestions = Column(Text)
    assessment_date = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    project = relationship("Project", back_populates="quality_assessments")


class DatabaseManager:
    """Database connection and session management."""
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or DATABASE_URL
        self.engine = None
        self.SessionLocal = None
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize database engine and session factory."""
        try:
            self.engine = create_engine(
                self.database_url,
                pool_pre_ping=True,
                pool_recycle=300,
                echo=False  # Set to True for SQL debugging
            )
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            logger.info("Database engine initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize database engine", error=str(e))
            raise DatabaseConnectionError(
                database_url=self.database_url,
                error_details=str(e)
            )
    
    def create_tables(self):
        """Create all database tables."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error("Failed to create database tables", error=str(e))
            raise DatabaseConnectionError(
                database_url=self.database_url,
                error_details=f"Table creation failed: {str(e)}"
            )
    
    def get_session(self) -> Session:
        """Get a database session."""
        if not self.SessionLocal:
            raise DatabaseConnectionError(
                database_url=self.database_url,
                error_details="Database not initialized"
            )
        return self.SessionLocal()
    
    def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error("Database health check failed", error=str(e))
            return False
    
    def close(self):
        """Close database connections."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connections closed")


# Global database manager instance
db_manager = DatabaseManager()


def get_db_session() -> Session:
    """Dependency function to get database session."""
    return db_manager.get_session()


def init_database():
    """Initialize database with tables and extensions."""
    try:
        # Create PostGIS extension if not exists
        with db_manager.get_session() as session:
            session.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
            session.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";")
            session.commit()
            logger.info("PostGIS extension enabled")
        
        # Create tables
        db_manager.create_tables()
        logger.info("Database initialization completed")
        
    except Exception as e:
        logger.error("Database initialization failed", error=str(e))
        raise


# Database utility functions

def create_project(
    session: Session,
    name: str,
    description: Optional[str] = None,
    village_code: Optional[str] = None,
    district: Optional[str] = None,
    state: Optional[str] = None
) -> Project:
    """Create a new project."""
    project = Project(
        name=name,
        description=description,
        village_code=village_code,
        district=district,
        state=state
    )
    session.add(project)
    session.commit()
    session.refresh(project)
    logger.info("Project created", project_id=str(project.id), name=name)
    return project


def get_project_by_id(session: Session, project_id: str) -> Optional[Project]:
    """Get project by ID."""
    return session.query(Project).filter(Project.id == project_id).first()


def get_active_ai_model(session: Session, model_type: str) -> Optional[AIModel]:
    """Get the active AI model for a given type."""
    return session.query(AIModel).filter(
        AIModel.model_type == model_type,
        AIModel.is_active == True
    ).first()


def create_point_cloud_record(
    session: Session,
    project_id: str,
    filename: str,
    file_size: int,
    point_count: int,
    coordinate_system: str,
    metadata: Optional[Dict[str, Any]] = None
) -> PointCloudDataset:
    """Create a point cloud dataset record."""
    point_cloud = PointCloudDataset(
        project_id=project_id,
        filename=filename,
        file_size=file_size,
        point_count=point_count,
        coordinate_system=coordinate_system,
        metadata=metadata or {}
    )
    session.add(point_cloud)
    session.commit()
    session.refresh(point_cloud)
    logger.info("Point cloud record created", point_cloud_id=str(point_cloud.id))
    return point_cloud


def update_processing_status(
    session: Session,
    record_id: str,
    table_class,
    status: str
):
    """Update processing status for any record."""
    record = session.query(table_class).filter(table_class.id == record_id).first()
    if record:
        record.processing_status = status
        record.updated_at = datetime.utcnow()
        session.commit()
        logger.info(
            "Processing status updated",
            record_id=record_id,
            table=table_class.__tablename__,
            status=status
        )


# Context manager for database sessions
class DatabaseSession:
    """Context manager for database sessions with automatic cleanup."""
    
    def __init__(self):
        self.session = None
    
    def __enter__(self) -> Session:
        self.session = db_manager.get_session()
        return self.session
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            if exc_type is not None:
                self.session.rollback()
                logger.error("Database session rolled back due to exception")
            else:
                self.session.commit()
            self.session.close()


# Initialize database on module import
if __name__ == "__main__":
    init_database()