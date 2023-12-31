import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import warnings

from flask import Flask, jsonify, request, render_template
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from flask_cors import CORS
from bson import json_util
import json
import re
import bcrypt
from search_and_recommend_files.recommend import recommend
from search_and_recommend_files.searchfirst import search1
import csv

app = Flask(__name__)
CORS(app)  # This will enable CORS for your entire app

uri = "mongodb+srv://dhruvjain657:NoOdK7bqKAJL0981@cluster0.bfryinm.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(uri, server_api=ServerApi('1')
                     )  # Change the URL as needed
# Replace 'your_database_name' with your actual database name
db = client['recommendationDB']
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


@app.route('/customer_cart')
def customer_cart():
    return render_template('customer_cart.html')


@app.route('/customer_index')
def customer_index():
    return render_template('customer_index.html')


@app.route('/customer_shop')
def customer_shop():
    return render_template('customer_shop.html')


@app.route('/customer_about')
def customer_about():
    return render_template('customer_about.html')


@app.route('/customer_contact')
def customer_contact():
    return render_template('customer_contact.html')


@app.route('/no_cart')
def no_cart():
    return render_template('no_cart.html')

@app.route('/customerindex')
def customerindex():
    return render_template('customerindex.html')

@app.route('/customeriindex')
def customeriindex():
    return render_template('customeriindex.html')


@app.route('/addproduct', methods=['POST'])
def add_data():
    data = request.json  # Assuming you're sending JSON data
    db.recommendationDB.insert_one(data)
    return jsonify({'message': 'Data added successfully'})


@app.route('/getproducts', methods=['GET'])
def get_data():
    data = list(db.recommendationDB.find())
    return jsonify(data)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        print(request.form.get('email'))
        # data = request.json
        email = request.form.get('email')
        password = request.form.get('password')

        # Hash the password before storing it
        hashed_password = bcrypt.hashpw(
            password.encode('utf-8'), bcrypt.gensalt())

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
        user = db.users.find({"email": email})
        print(list(user))

        # Check if the username is already taken
        if not user:
            return render_template('login.html')
        else:
            return render_template('customer_index.html', user=user)
    return render_template('login.html')


@app.route('/search', methods=['POST'])
def search_data():
    search_str = request.json["productName"]
    recommendation = search1(search_str)

    with open('search_and_recommend_files/df_mean.csv', 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        products = []

        for product in recommendation:
            for row in csv_reader:
                if row['product_title'] == product:
                    products.append(row)
            csv_file.seek(0)
  
    return products

if __name__ == '__main__':
    app.run(debug=True)
