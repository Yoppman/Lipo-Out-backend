# Lipo-Out-backend
### Introduction
Lipo-Out is a fitness app that track user's diet and give incredible advices and planning.


# Backend API Documentation

## Overview
This is a FastAPI-based backend application that provides two main entities: `User` and `Food`. The API allows users to create, read, update, and delete records for both entities. The relationship between the two tables is that a `User` can have multiple `Food` entries.

The database used in this application is SQLite, and it is managed using `SQLModel`, which is a combination of SQLAlchemy and Pydantic.

## Setup Instructions

### Prerequisites
- Python 3.7+
- FastAPI
- SQLModel
- SQLite
- Uvicorn (to run the FastAPI application)

### Installing Dependencies
The first step is download the poetry[https://python-poetry.org/].

Check the version first and init.
```bash
poetry --version
```

Initialization, just follow the configuration and it will generate a `pyproject.toml`
```bash
poetry init
```

Install the dependencies using `poetry` or `pip` (here we use peotry):
```bash
poetry add fastapi sqlmodel uvicorn
```

To activate the virtual environment
```bash
poetry shell
```

### Running the Application
To run the FastAPI application, use the following command:
```bash
uvicorn main:app --reload
```
Here `main` refers to the file name of your FastAPI app. If your file is named differently, adjust this command accordingly.

The application will be available at `http://127.0.0.1:8000/`.

### Database Initialization
The database is automatically created upon startup. When you run the application, it will generate `database.db` (SQLite file) with the required tables.

## API Endpoints

### User Endpoints

#### 1. Create a New User
**POST** `/users/`

- **Request Body**:
  ```json
  {
    "name": "John Doe",
    "age": 30,
    "height": 180,
    "weight": 75,
    "goal": "Lose weight"
  }
  ```

- **Response**:
  ```json
  {
    "id": 1,
    "name": "John Doe",
    "age": 30,
    "height": 180,
    "weight": 75
  }
  ```

#### 2. Get All Users
**GET** `/users/`

- **Query Parameters** (optional):
  - `offset` (default: 0)
  - `limit` (default: 100, max: 100)

- **Response**:
  ```json
  [
    {
      "id": 1,
      "name": "John Doe",
      "age": 30,
      "height": 180,
      "weight": 75
    }
  ]
  ```

#### 3. Get a Specific User by ID
**GET** `/users/{user_id}`

- **Response**:
  ```json
  {
    "id": 1,
    "name": "John Doe",
    "age": 30,
    "height": 180,
    "weight": 75
  }
  ```

#### 4. Update a User
**PATCH** `/users/{user_id}`

- **Request Body** (you can provide any fields you want to update):
  ```json
  {
    "name": "Jane Doe",
    "weight": 70
  }
  ```

- **Response**:
  ```json
  {
    "id": 1,
    "name": "Jane Doe",
    "age": 30,
    "height": 180,
    "weight": 70
  }
  ```

#### 5. Delete a User
**DELETE** `/users/{user_id}`

- **Response**:
  ```json
  {
    "ok": true
  }
  ```

### Food Endpoints

#### 1. Create a New Food Entry
**POST** `/foods/`

- **Request Body**:
  ```json
  {
    "food_analysis": "High in protein",
    "food_photo": "binary_data_here",
    "user_id": 1
  }
  ```

- **Response**:
  ```json
  {
    "food_id": 1,
    "food_analysis": "High in protein",
    "user_id": 1
  }
  ```

#### 2. Get All Food Entries
**GET** `/foods/`

- **Query Parameters** (optional):
  - `offset` (default: 0)
  - `limit` (default: 100, max: 100)

- **Response**:
  ```json
  [
    {
      "food_id": 1,
      "food_analysis": "High in protein",
      "user_id": 1
    }
  ]
  ```

#### 3. Get a Specific Food Entry by ID
**GET** `/foods/{food_id}`

- **Response**:
  ```json
  {
    "food_id": 1,
    "food_analysis": "High in protein",
    "user_id": 1
  }
  ```

#### 4. Update a Food Entry
**PATCH** `/foods/{food_id}`

- **Request Body** (you can provide any fields you want to update):
  ```json
  {
    "food_analysis": "Low carb",
    "food_photo": "new_binary_data_here"
  }
  ```

- **Response**:
  ```json
  {
    "food_id": 1,
    "food_analysis": "Low carb",
    "user_id": 1
  }
  ```

#### 5. Delete a Food Entry
**DELETE** `/foods/{food_id}`

- **Response**:
  ```json
  {
    "ok": true
  }
  ```

## Database Schema

### User Table
| Field  | Type  | Description              |
|--------|-------|--------------------------|
| id     | int   | Primary key (auto-generated) |
| name   | str   | User's name              |
| age    | int   | User's age (optional)    |
| height | int   | User's height (optional) |
| weight | int   | User's weight (optional) |
| goal   | str   | User's fitness goal      |

### Food Table
| Field        | Type  | Description                    |
|--------------|-------|--------------------------------|
| food_id      | int   | Primary key (auto-generated)   |
| food_analysis| str   | Analysis of the food item      |
| food_photo   | bytes | Binary data of the food photo  |
| user_id      | int   | Foreign key linking to `User`  |

## Error Handling
The API returns appropriate HTTP error codes and messages when something goes wrong:
- **400 Bad Request**: When a required field is missing or invalid (e.g., invalid `user_id`).
- **404 Not Found**: When the requested resource (user or food) does not exist.
- **500 Internal Server Error**: For unexpected errors within the server.

## Conclusion
This backend provides a simple yet robust way to manage users and their associated food entries. The API uses FastAPI's high-performance capabilities to handle requests efficiently, and the database is managed using SQLModel to ensure data integrity and ease of querying.

