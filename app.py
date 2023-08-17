from flask import Flask, jsonify, request, render_template
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from flask_cors import CORS
from bson import json_util
import re, bcrypt

app = Flask(__name__)
CORS(app)  # This will enable CORS for your entire app

uri = "mongodb+srv://dhruvjain657:NoOdK7bqKAJL0981@cluster0.bfryinm.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(uri, server_api=ServerApi('1'))  # Change the URL as needed
db = client['recommendationDB']  # Replace 'your_database_name' with your actual database name
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/shop')
def shop():
    return render_template('shop.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/cart')
def cart():
    return render_template('cart.html')
    
@app.route('/addproduct', methods=['POST'])
def add_data():
    data = request.json  # Assuming you're sending JSON data
    db.recommendationDB.insert_one(data)
    return jsonify({'message': 'Data added successfully'})

@app.route('/getproducts', methods=['GET'])
def get_data():
    data = list(db.recommendationDB.find())
    return json_util.dumps(data)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        print(request.form.get('email'))
        # data = request.json
        email = request.form.get('email')
        password = request.form.get('password')

        # Hash the password before storing it
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        # Check if the username is already taken
        if db.users.find_one({"email": email}):
            return jsonify({"message": "Email already taken"}), 400

        # Insert the user into the database
        db.users.insert_one({
            "email": email,
            "confirmEmail": request.form.get("confirmEmail"),
            "password": hashed_password,
            "fullName": request.form.get("fullname"),
            "street": request.form.get("street"),
            "postal": request.form.get("postal"),
            "city": request.form.get("city")
        })

        return render_template('login.html')
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # data = request.json
        email = request.form.get('email')
        password = request.form.get('password')

        # Hash the password before storing it
        # hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())    
        user = json_util.dumps(db.users.find({"email": email}))
        
        # Check if the username is already taken
        if not user:
            return render_template('login.html')
        else:
            return render_template('index.html', user=user)
    return render_template('login.html')

@app.route('/search', methods=['POST'])
def search_data():
    search_str = request.json["productName"]  # Assuming you're sending JSON data
    # Create a regex pattern for searching similar product names
    regex_pattern = re.compile(f".*{search_str}.*", re.IGNORECASE)

    data  = db.recommendationDB.find({ "productName": regex_pattern })
    return json_util.dumps(data)

if __name__ == '__main__':
    app.run(debug=True)