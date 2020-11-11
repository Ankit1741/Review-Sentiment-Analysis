from flask import Flask,render_template,url_for,request
import pickle
import re





app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

vect =  pickle.load(open('vectorizer.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('mini_project.html')

@app.route('/predict', methods=['POST'])

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def text_preprocess(text):
    text = text.lower() #converts to lower alphabets 
    #removing HTML tags
    cleanr = re.compile('<.*?>')
    text = re.sub(cleanr, '', text)
    
    #decontract words ex: 'haven't' -> 'have not' etc
    text = decontracted(text)
    
    #removing digits
    for word in text:
        if word.isdigit():
            text = text.replace(word, ' ')
    
    #removing words of length higher or equal to 15
    
    text = re.sub(r'\b\w{15,}', '', text)
    return text

def predict():
    if request.method == 'POST':
        
        message = request.form['message']
        
        review = message
        
        cleaned_review = text_preprocess(review)
        
        data = vect.transform(cleaned_review)
        
        y_hat = model.predict(data)
        
       

        

            
    return render_template('mini_project.html', prediction=y_hat)
    
if __name__=="__main__":
    app.run(debug=True)
        