# Egyptian Hieroglyphs Django Web App

This Django project provides a web interface for translating Egyptian hieroglyph images and chatting about Ancient Egypt. It includes user authentication, image upload and translation, and a chatbot interface.

## Features

- **User Authentication:** Sign up and log in to access features.
- **Image Upload & Translation:** Upload images of hieroglyphs and receive translations, stories, and symbol breakdowns.
- **Chatbot:** Ask questions about Ancient Egypt and receive answers from an integrated chatbot.
- **Modern UI:** Uses Django templates and static files for a user-friendly experience.

## Project Structure

- `myproject/` — Django project settings and URLs.
- `myapp/` — Main application with models, views, forms, templates, and static files.
- `manage.py` — Django management script.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Apply migrations:**
   ```bash
   python manage.py migrate
   ```

3. **Create a superuser (optional, for admin access):**
   ```bash
   python manage.py createsuperuser
   ```

4. **Run the development server:**
   ```bash
   python manage.py runserver
   ```

5. **Access the app:**
   - Visit [http://localhost:8000/](http://localhost:8000/) in your browser.

## Usage

- **Home:** Landing page after login.
- **Translator:** Upload an image of hieroglyphs to get a translation and symbol analysis.
- **Chatbot:** Ask questions about Ancient Egypt.
- **Login/Signup:** Register or log in to use the app.

## Models

- `UploadedImage`: Stores uploaded images and timestamps.
- `Chatbot`: Stores user questions and bot responses.

## API Integration

- The translator view sends images to an external API at `http://localhost:8000/translate`.
- The chatbot view sends user queries to an API at `http://localhost:8080/chat`.

## Static & Media Files

- Static files (CSS, JS, images) are in `myapp/static/`.
- Uploaded images are stored in the media directory (configure in settings if needed).

## Notes

- Make sure the external translation and chatbot APIs are running and accessible at the specified URLs.
- Do not commit `db.sqlite3`, `.pyc` files, or `__pycache__` directories to version control.
