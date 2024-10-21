from typing import Annotated
from fastapi import Depends, FastAPI, HTTPException, Query
from sqlmodel import Field, Session, SQLModel, create_engine, select, Relationship

# Define the UserBase and User classes as they are now
class UserBase(SQLModel):
    name: str = Field(index=True)
    age: int | None = Field(default=None, index=True)
    height: int | None = Field(default=None, index=True)
    weight: int | None = Field(default=None, index=True)


class User(UserBase, table=True):
    id: int | None = Field(default=None, primary_key=True)
    goal: str

    # Relationship to Food table
    foods: list["Food"] = Relationship(back_populates="user")


class UserPublic(UserBase):
    id: int


class UserCreate(UserBase):
    goal: str


class UserUpdate(UserBase):
    name: str | None = None
    age: int | None = None
    weight: str | None = None
    height: str | None = None
    goal: str | None = None


# Food table classes similar to the User table

class FoodBase(SQLModel):
    food_analysis: str  # Common field to all food-related classes
    food_photo: bytes  # Store the image as binary data


class Food(FoodBase, table=True):
    food_id: int | None = Field(default=None, primary_key=True)  # Automatically generated
    user_id: int | None = Field(default=None, foreign_key="user.id")  # Foreign key to User table

    # Relationship back to User
    user: "User" = Relationship(back_populates="foods")


class FoodPublic(FoodBase):
    food_id: int  # Exposing the food ID to the client
    user_id: int  # Exposing the user ID to the client


class FoodCreate(FoodBase):
    food_photo: bytes  # Must provide the food photo when creating
    user_id: int


class FoodUpdate(FoodBase):
    food_photo: bytes | None = None  # Optional for updates
    food_analysis: str | None = None  # Optional for updates


sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]
app = FastAPI()


@app.on_event("startup")
def on_startup():
    create_db_and_tables()

# User endpoints (no changes)
@app.post("/users/", response_model=UserPublic)
def create_user(user: UserCreate, session: SessionDep):
    db_user = User.model_validate(user)
    session.add(db_user)
    session.commit()
    session.refresh(db_user)
    return db_user


@app.get("/users/", response_model=list[UserPublic])
def read_users(
    session: SessionDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=100)] = 100,
):
    users = session.exec(select(User).offset(offset).limit(limit)).all()
    return users


@app.get("/users/{user_id}", response_model=UserPublic)
def read_user(user_id: int, session: SessionDep):
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.patch("/users/{user_id}", response_model=UserPublic)
def update_user(user_id: int, user: UserUpdate, session: SessionDep):
    user_db = session.get(User, user_id)
    if not user_db:
        raise HTTPException(status_code=404, detail="User not found")
    user_data = user.model_dump(exclude_unset=True)
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


# New Food endpoints following the same architecture as the User table

@app.post("/foods/", response_model=FoodPublic)
def create_food(food: FoodCreate, session: SessionDep):
    # Ensure the user_id is valid and the user exists
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
    food = session.get(Food, food_id)
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