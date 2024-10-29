from app import create_app

# Initialize the Flask app using the factory function
app = create_app()

if __name__ == '__main__':
    # Run the app locally with debug mode enabled
    app.run(debug=True)
