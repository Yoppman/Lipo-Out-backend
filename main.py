from typing import Annotated
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Query,
    status,
    Path
)
from sqlmodel import (
    Field,
    Session,
    SQLModel,
    create_engine,
    select,
    Relationship,
)
from sqlalchemy.orm import joinedload
from typing import Optional

# Database configuration
sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"
connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)

# Database session dependency
def get_session():
    with Session(engine) as session:
        yield session

SessionDep = Annotated[Session, Depends(get_session)]

# Initialize database
def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

# FastAPI app setup
app = FastAPI()

@app.on_event("startup")
def on_startup():
    create_db_and_tables()

# User models
class UserBase(SQLModel):
    name: str = Field(index=True)
    age: int | None = Field(default=None, index=True)
    height: int | None = Field(default=None, index=True)
    weight: int | None = Field(default=None, index=True)
    goal: str | None = Field(default=None, index=True)
    telegram_id: int | None = Field(index=True)  # Optional field for Telegram user ID
    # Add other platform-specific IDs here if needed, e.g., `facebook_id`, `whatsapp_id`

class User(UserBase, table=True):
    id: int | None = Field(default=None, primary_key=True)
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
    telegram_id: Optional[int] = None

# Food models
class FoodBase(SQLModel):
    food_analysis: str
    food_photo: bytes
    protein: float | None = Field(default=0.0, description="Protein content in grams")
    carb: float | None = Field(default=0.0, description="Carbohydrate content in grams")
    fat: float | None = Field(default=0.0, description="Fat content in grams")
    calories: float | None = Field(default=0.0, description="Caloric content")

class Food(FoodBase, table=True):
    food_id: int | None = Field(default=None, primary_key=True)
    user_id: int | None = Field(default=None, foreign_key="user.id")
    user: "User" = Relationship(back_populates="foods")

class FoodPublic(FoodBase):
    food_id: int
    user_id: int

class FoodCreate(FoodBase):
    food_photo: bytes
    user_id: int

class FoodUpdate(FoodBase):
    food_photo: bytes | None = None
    food_analysis: str | None = None

# User endpoints
@app.post("/users/", response_model=UserPublic, status_code=status.HTTP_201_CREATED)
def create_user(user: UserCreate, session: SessionDep):
    db_user = User.model_validate(user)
    session.add(db_user)
    session.commit()
    session.refresh(db_user)
    return db_user

@app.get("/users/", response_model=list[UserPublic])
def read_users(
    session: SessionDep,
    name: Optional[str] = Query(None),  # Optional name query parameter
    telegram_id: Optional[int] = Query(None),  # Optional telegram_id query parameter
    offset: int = 0,
    limit: Annotated[int, Query(le=100)] = 100,
):
    # Query with conditional filtering
    if name:
        # If a name is provided, filter by name
        query = select(User).where(User.name == name)
    elif telegram_id:
        # If a telegram_id is provided, filter by telegram_id
        query = select(User).where(User.telegram_id == telegram_id)
    else:
        # If no filters are provided, return all users with pagination
        query = select(User).offset(offset).limit(limit)

    # Execute the query and fetch results
    users = session.exec(query).all()
    
    if not users:
        raise HTTPException(status_code=404, detail="User not found")
    
    return users

@app.get("/users/{user_id}", response_model=UserPublic)
def read_user(user_id: int, session: SessionDep):
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.patch("/users/", response_model=UserPublic)
def update_user(
    user: UserUpdate,  # `UserUpdate` model with optional fields
    session: SessionDep,
    user_id: Optional[int] = None,  # Optional `user_id`
    telegram_id: Optional[int] = Query(None, description="The Telegram ID"),  # Optional `telegram_id`
):
    # Query by `telegram_id` if provided, otherwise by `user_id`
    user_db = None
    if telegram_id is not None:
        user_db = session.exec(select(User).where(User.telegram_id == telegram_id)).first()
    elif user_id is not None:
        user_db = session.get(User, user_id)
    
    # Check if user was found
    if not user_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Update only the fields provided in the `user` object
    user_data = user.model_dump(exclude_unset=True)
    
    # Apply the updates
    user_db.sqlmodel_update(user_data)
    session.add(user_db)
    session.commit()
    session.refresh(user_db)
    
    return user_db

@app.delete("/users/{user_id}")
def delete_user(user_id: int, session: SessionDep):
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    session.delete(user)
    session.commit()
    return {"ok": True}

# Food endpoints
@app.post("/foods/", response_model=FoodPublic)
def create_food(food: FoodCreate, session: SessionDep):
    db_user = session.get(User, food.user_id)
    if not db_user:
        raise HTTPException(status_code=400, detail="User not found")

    db_food = Food.model_validate(food)
    session.add(db_food)
    session.commit()
    session.refresh(db_food)
    return db_food

@app.get("/foods/", response_model=list[FoodPublic])
def read_foods(
    session: SessionDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=100)] = 100,
):
    foods = session.exec(select(Food).offset(offset).limit(limit)).all()
    return foods

@app.get("/foods/{food_id}", response_model=FoodPublic)
def read_food(food_id: int, session: SessionDep):
    query = select(Food).where(Food.food_id == food_id).options(joinedload(Food.user))
    food = session.exec(query).first()
    if not food:
        raise HTTPException(status_code=404, detail="Food not found")
    return food

@app.patch("/foods/{food_id}", response_model=FoodPublic)
def update_food(food_id: int, food: FoodUpdate, session: SessionDep):
    food_db = session.get(Food, food_id)
    if not food_db:
        raise HTTPException(status_code=404, detail="Food not found")
    food_data = food.model_dump(exclude_unset=True)
    food_db.sqlmodel_update(food_data)
    session.add(food_db)
    session.commit()
    session.refresh(food_db)
    return food_db

@app.delete("/foods/{food_id}")
def delete_food(food_id: int, session: SessionDep):
    food = session.get(Food, food_id)
    if not food:
        raise HTTPException(status_code=404, detail="Food not found")
    session.delete(food)
    session.commit()
    return {"ok": True}