from flask import Flask, render_template
import gzip
import json
from flask import  request, jsonify
from random import sample
data_list = [json.loads(l) for l in gzip.open("ds1000.jsonl.gz", "rt").readlines()]

app = Flask(__name__)

# Sample list of dictionaries


@app.route('/')
def index():
    # Display the first dictionary in the list
    first_data = data_list[0]
    return render_template('index3.html', data=first_data)

@app.route('/get_data', methods=['POST'])
def get_data():
    selected_tag = request.json['tag']
    print(selected_tag)
    data_with_tag = sample([data for data in data_list if selected_tag == data['metadata']['library'].lower()],1)[0]
    return jsonify(data_with_tag)

if __name__ == '__main__':
    app.run(debug=True)