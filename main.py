import logging
from typing import Optional, Annotated
from fastapi import FastAPI, HTTPException, Query, status, Depends
from contextlib import asynccontextmanager
from sqlmodel import SQLModel, Field, select, Relationship
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import BigInteger, Column
from datetime import datetime
from sqlalchemy import DateTime
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import logging

logger = logging.getLogger(__name__)

load_dotenv()

postgres_url = os.getenv('POSTGRES_URL')  # Or 'POSTGRES_URL' if using an internal URL
postgres_url = "postgresql://postgres:leZfVJuoupqiTPbTcizIIbbpfsggAUII@junction.proxy.rlwy.net:40673/railway"
# Ensure the URL is prefixed correctly
if 'asyncpg' not in postgres_url:
    postgres_url = postgres_url.replace('postgresql://', 'postgresql+asyncpg://')
# Create async engine
from sqlalchemy.ext.asyncio import create_async_engine

engine = create_async_engine(postgres_url, echo=True, future=True)
async_session_maker = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# Dependency for the async session
async def get_session() -> AsyncSession:
    async with async_session_maker() as session:
        yield session

# Annotated type for FastAPI dependency injection
SessionDep = Annotated[AsyncSession, Depends(get_session)]

# when database exists
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application...")
    yield
    logger.info("Shutting down application...")

origins = [
    "*",  # Allows requests from any origin
    # Add specific origins if needed
]

app = FastAPI(lifespan=lifespan)

# Add CORS middleware to allow requests from specified origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)


# # when init database
# app = FastAPI()

# @app.on_event("startup")
# async def on_startup():
#     async with engine.begin() as conn:
#         await conn.run_sync(SQLModel.metadata.create_all)

# Models
class UserBase(SQLModel):
    name: str = Field(index=True)
    age: Optional[int] = Field(default=None, index=True)
    height: Optional[int] = Field(default=None, index=True)
    weight: Optional[int] = Field(default=None, index=True)
    goal: Optional[str] = Field(default=None, index=True)
    telegram_id: Optional[int] = Field(default=None, sa_column=Column(BigInteger, index=True))


class User(UserBase, table=True):
    __table_args__ = {"extend_existing": True}
    __tablename__ = 'user'  # Ensure table name is consistent
    id: Optional[int] = Field(default=None, primary_key=True)
    goal: str
    foods: list["Food"] = Relationship(back_populates="user")

class UserPublic(UserBase):
    id: int

class UserCreate(UserBase):
    goal: str

class UserUpdate(UserBase):
    name: Optional[str] = None
    age: Optional[int] = None
    weight: Optional[int] = None
    height: Optional[int] = None
    goal: Optional[str] = None
    telegram_id: Optional[int] = Field(default=None, sa_column=Column(BigInteger, index=True))

class FoodBase(SQLModel):
    food_analysis: str
    food_photo: bytes
    protein: Optional[float] = Field(default=0.0, description="Protein content in grams")
    carb: Optional[float] = Field(default=0.0, description="Carbohydrate content in grams")
    fat: Optional[float] = Field(default=0.0, description="Fat content in grams")
    calories: Optional[float] = Field(default=0.0, description="Caloric content")

# Updated Food model to include the date column
class Food(FoodBase, table=True):
    __table_args__ = {"extend_existing": True}
    food_id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[int] = Field(default=None, foreign_key="user.id")
    date: datetime = Field(default_factory=datetime.utcnow)  # New date column with default timestamp
    user: User = Relationship(back_populates="foods")

# Updated FoodPublic and FoodCreate to include the date field
class FoodPublic(FoodBase):
    food_id: int
    user_id: int
    date: Optional[datetime] = None  # Optional date field for food creation

class FoodCreate(FoodBase):
    food_photo: bytes
    user_id: int
    date: Optional[datetime] = None  # Make date optional

# FoodUpdate model for updating food entries
class FoodUpdate(FoodBase):
    food_analysis: Optional[str] = None
    food_photo: Optional[bytes] = None
    protein: Optional[float] = None
    carb: Optional[float] = None
    fat: Optional[float] = None
    calories: Optional[float] = None

# Endpoints
@app.post("/users/", response_model=UserPublic, status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate, session: SessionDep):
    db_user = User(**user.dict())
    session.add(db_user)
    await session.commit()
    await session.refresh(db_user)
    return db_user

@app.get("/users/", response_model=list[UserPublic])
async def read_users(
    session: SessionDep,
    name: Optional[str] = Query(None),  # Optional name query parameter
    telegram_id: Optional[int] = Query(None),  # Optional telegram_id query parameter
    offset: int = 0,
    limit: Annotated[int, Query(le=100)] = 100,
):
    # Build the query with conditional filtering
    if name:
        query = select(User).where(User.name == name)
    elif telegram_id:
        query = select(User).where(User.telegram_id == telegram_id)
    else:
        query = select(User).offset(offset).limit(limit)

    # Execute the query and fetch results
    result = await session.execute(query)
    users = result.scalars().all()
    
    if not users:
        raise HTTPException(status_code=404, detail="User not found")
    
    return users

@app.get("/users/{user_id}", response_model=UserPublic)
async def read_user(user_id: int, session: SessionDep):
    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.patch("/users/", response_model=UserPublic)
async def update_user(
    user: UserUpdate,  # `UserUpdate` model with optional fields
    session: SessionDep,
    user_id: Optional[int] = None,  # Optional `user_id`
    telegram_id: Optional[int] = Query(None, description="The Telegram ID"),  # Optional `telegram_id`
):
    # Query by `telegram_id` if provided, otherwise by `user_id`
    user_db = None
    if telegram_id is not None:
        result = await session.execute(select(User).where(User.telegram_id == telegram_id))
        user_db = result.scalars().first()
    elif user_id is not None:
        user_db = await session.get(User, user_id)
    
    # Check if user was found
    if not user_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Update only the fields provided in the `user` object
    user_data = user.model_dump(exclude_unset=True)
    
    # Apply the updates
    for key, value in user_data.items():
        setattr(user_db, key, value)
    
    session.add(user_db)
    await session.commit()
    await session.refresh(user_db)
    
    return user_db

# Delete user endpoint
@app.delete("/users/{user_id}", response_model=dict)
async def delete_user(user_id: int, session: SessionDep):
    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    await session.delete(user)
    await session.commit()
    return {"detail": "User deleted successfully"}

@app.post("/foods/", response_model=FoodPublic)
async def create_food(food: FoodCreate, session: SessionDep):
    # Check if the user exists
    db_user = await session.get(User, food.user_id)
    if not db_user:
        raise HTTPException(status_code=400, detail="User not found")

    # Prepare the food data, making `date` naive if it’s provided
    food_data = food.dict(exclude_unset=True)

    # Convert `date` to naive format if it exists
    if food_data.get("date"):
        food_data["date"] = food_data["date"].replace(tzinfo=None)

    # Create the Food instance
    db_food = Food(**food_data)
    session.add(db_food)
    await session.commit()
    await session.refresh(db_food)
    return db_food

@app.get("/foods/", response_model=list[FoodPublic])
async def read_foods(
    session: SessionDep, 
    user_id: Optional[int] = Query(None, description="ID of the user to filter foods by"), 
    date: Optional[datetime] = Query(None, description="Retrieve foods uploaded after this date"),
    offset: int = 0, 
    limit: int = 100
):
    # Build the query based on the presence of `user_id` and `date`
    query = select(Food)
    
    # Apply filters based on `user_id` and `date`
    if user_id:
        query = query.where(Food.user_id == user_id)
    if date:
        query = query.where(Food.date > date)
    
    # Apply pagination
    query = query.offset(offset).limit(limit)
    
    # Execute the query and retrieve results
    result = await session.execute(query)
    foods = result.scalars().all()
    
    # Check if no foods found for the user or date range
    if (user_id or date) and not foods:
        raise HTTPException(status_code=404, detail="No foods found for the specified filters")
    
    return foods

@app.get("/foods/{food_id}", response_model=FoodPublic)
async def get_food_by_id(food_id: int, session: SessionDep):
    food = await session.get(Food, food_id)
    if not food:
        raise HTTPException(status_code=404, detail="Food not found")
    return food

# Update food endpoint
@app.patch("/foods/{food_id}", response_model=FoodPublic)
async def update_food(food_id: int, food: FoodUpdate, session: SessionDep):
    food_db = await session.get(Food, food_id)
    if not food_db:
        raise HTTPException(status_code=404, detail="Food not found")
    
    # Update only the fields provided in the `food` object
    food_data = food.model_dump(exclude_unset=True)
    
    for key, value in food_data.items():
        setattr(food_db, key, value)
    
    session.add(food_db)
    await session.commit()
    await session.refresh(food_db)
    
    return food_db

# Delete food endpoint
@app.delete("/foods/{food_id}", response_model=dict)
async def delete_food(food_id: int, session: SessionDep):
    food = await session.get(Food, food_id)
    if not food:
        raise HTTPException(status_code=404, detail="Food not found")
    
    await session.delete(food)
    await session.commit()
    return {"detail": "Food deleted successfully"}


# Add a root endpoint to check if the server is running
@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "API is running"}