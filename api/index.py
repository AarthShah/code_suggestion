"""
Vercel entrypoint for Flask application
"""
from web_app import app

# Vercel expects the Flask app to be available as 'app'
# The web_app.py file already defines 'app', so we just import it
