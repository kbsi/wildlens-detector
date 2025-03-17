# Wildlife Footprint Identifier

## Overview

The Wildlife Footprint Identifier is a mobile application designed to help users identify wild animal footprints from photos taken with their devices. The application leverages a machine learning model to classify footprints and provide information about the species.

## Project Structure

The project is organized into three main components:

- **Frontend**: A React Native application that provides the user interface.
- **Backend**: A FastAPI server that handles API requests and interacts with the machine learning model.
- **Machine Learning Model**: A collection of scripts and notebooks for training and inference of the footprint classification model.

## Frontend

The frontend is built using React Native and includes the following key components:

- **Screens**: Different screens for capturing photos, displaying results, and showing history.
- **Components**: Reusable components such as camera integration and result views.
- **Services**: Functions for API calls and local storage management.

### Key Files

- `App.tsx`: Main entry point of the application.
- `CameraScreen.tsx`: Screen for capturing photos.
- `ResultScreen.tsx`: Screen for displaying identification results.

## Backend

The backend is built using FastAPI and provides endpoints for handling requests related to footprint identification. It includes:

- **API Endpoints**: For uploading photos and retrieving species information.
- **Database Models**: For storing footprint and user data.
- **Services**: Business logic for processing footprint data.

### Key Files

- `main.py`: Entry point for the FastAPI application.
- `footprints.py`: API endpoints for footprint-related requests.

## Machine Learning Model

The machine learning model is responsible for classifying the footprints based on the images provided. It includes:

- **Data Preprocessing**: Scripts for preparing the dataset.
- **Model Training**: Logic for training the classification model.
- **Inference**: Functions for making predictions with the trained model.

### Key Files

- `train.py`: Script for training the model.
- `predict.py`: Script for making predictions.

## Setup Instructions

1. **Clone the repository**:

   ```
   git clone <repository-url>
   cd wildlife-footprint-identifier
   ```

2. **Frontend Setup**:

- Navigate to the `frontend` directory.
- Install dependencies:
  ```
  npm install
  ```
- Run the application:
  ```
  npm start
  ```

3. **Backend Setup**:

- Navigate to the `backend` directory.
- Install pipenv if not already installed:
  ```
  pip install pipenv
  ```
- Install dependencies and activate virtual environment:
  ```
  pipenv install
  pipenv shell
  ```
- Run the FastAPI server:
  ```
  uvicorn app.main:app --reload
  ```

4. **Machine Learning Model Setup**:

- Navigate to the `ml_model` directory.
- Install pipenv if not already installed:
  ```
  pip install pipenv
  ```
- Install dependencies and activate virtual environment:
  ```
  pipenv install
  pipenv shell
  ```

## Usage

- Open the application on your mobile device.
- Use the camera feature to capture a footprint.
- View the identification results and additional information about the species.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
