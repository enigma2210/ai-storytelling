
# Rishabh's AI_based Storyteller

Welcome to the ultimate AI-powered storytelling platform.

## Features

- Robust text and audio prompt input
- Cutting-edge AI story generation with Gemini
- Detailed character profiles and stunning images
- AI-generated background scenes and composite artworks
- Smooth UI with advanced animations and responsiveness

## Prerequisites

- Python 3.8+
- FFmpeg (command line tool, required for audio processing)
- API keys:
   - Google Gemini API
   - Hugging Face API

## Setup Instructions

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/yourrepo.git
   cd yourrepo
   ```
2. Setup virtual environment:

   - Using venv:
     ```bash
     python -m venv env
     source env/bin/activate  # Mac/Linux
     env\Scripts\activate  # Windows
     ```

   - Using conda:
     ```bash
     conda create -n storyteller python=3.9 -y
     conda activate storyteller
     ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Install FFmpeg:

   - Ubuntu/Debian:
     ```bash
     sudo apt update
     sudo apt install ffmpeg
     ```

   - macOS:
     ```bash
     brew install ffmpeg
     ```

   - Windows:
     Download from:
     https://ffmpeg.org/download.html
     Add the bin directory to your PATH environment variable.

5. Set environment variables:

Create a `.env` file in the root:
```
GOOGLE_API_KEY=your_google_gemini_api_key
HUGGINGFACE_API_KEY=your_hf_api_key
SECRET_KEY=your_django_secret_key
DEBUG=True
ALLOWED_HOSTS=127.0.0.1,localhost
```

6. Run migrations:
   ```bash
   python manage.py migrate
   ```
7. Start development server:
   ```bash
   python manage.py runserver
   ```
8. Access the app:

Open browser at [http://localhost:8000](http://localhost:8000)

## Usage Guide

- Input text or upload audio
- Click Generate
- Wait for generated story and images
- Download or share outputs

## Troubleshooting

- Ensure FFmpeg is installed and in PATH
- Validate API keys and quotas
- Check logs for errors
- For port conflicts, run: `python manage.py runserver 8001`

## Development Workflow

- Modify code
- Commit changes
- Push to repository
- Test locally

## Deployment

- Prepare for production with env vars and static files
- Consider using platforms like Heroku, AWS, or Azure

## Contribution

Contributions welcome! Fork and submit pull requests.

## License

Licensed under MIT License

## Contact

Made with love by the developer.

