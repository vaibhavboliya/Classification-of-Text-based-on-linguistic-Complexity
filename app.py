# from urllib import request
from flask import Flask, render_template, request, redirect
from jinja2 import Environment, PackageLoader, select_autoescape
from flask_pymongo import PyMongo
import joblib
import string
import dill
import lime
import lime.lime_tabular
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from nltk.corpus import stopwords
app = Flask(__name__, static_url_path='/static')
app.config["MONGO_URI"] = "mongodb://localhost:27017/VAERS_Data"
mongo = PyMongo(app)
loaded_rf = joblib.load("rb_boost.joblib")
loaded_tfidf_vect = joblib.load("tfidf_vect.joblib")
nltk.download('stopwords')


def preprocessing(text):
    stop = stopwords.words('english')
    string.punctuation
    clean_txt = "".join([i for i in text if i not in string.punctuation])
    clean_txt = clean_txt.lower()
    clean_txt = clean_txt.replace('\d+', '')
    l = clean_txt.split(" ")
    final = []
    f = " "
    for word in l:
        if word not in (stop):
            final.append(word)
    clean_txt = f.join(final)
    return clean_txt


@app.route("/")
def index():
    return redirect('/form')
    data5 = mongo.db.dead_data.find()
    data4 = mongo.db.hospital_data.find()
    data = mongo.db.Daily_ADRS_count.find().sort("date")
    data1 = mongo.db.gender_distribution_data.find()
    data2 = mongo.db.adrs_distribution.find()
    data3 = mongo.db.symptoms_data.find().sort("date")
    count = 0
    symptoms_label = []
    symptoms_percentage = []
    status_hospital = []
    status_hospital_percentage = []
    status_dead = []
    status_dead_percentage = []
    for x in data5:
        status_dead.append(x["Status"])
        status_dead_percentage.append(x["Values_in_percentage"])

    for x in data4:
        status_hospital.append(x["Status"])
        status_hospital_percentage.append(x["Values_in_percentage"])
    for x in data3:
        if count == 10:
            break
        count = count + 1
        symptoms_label.append(x["data"])
        z = int(x["percentage"][:2])
        symptoms_percentage.append("width:"+str(z)+"%")
    adrs = []
    count_adrs = []
    gender_label = []
    gender_percentage = []
    date = []
    count = []
    for x in data2:
        adrs.append(x["label"])
        count_adrs.append(x["data"])
    for x in data1:
        gender_label.append(x["Gender"])
        gender_percentage.append(x["Values_in_percentage"])
    for x in data:
        date.append(x["date"])
        count.append(x["count"])
    return render_template("index.html",
                           date=date, count=count, gender_label=gender_label, gender_percentage=gender_percentage, adrs=adrs, count_adrs=count_adrs,
                           symptoms_label=symptoms_label, symptoms_percentage=symptoms_percentage,
                           status_dead=status_dead, status_dead_percentage=status_dead_percentage,
                           status_hospital=status_hospital, status_hospital_percentage=status_hospital_percentage)


@app.route("/profile")
def profile():
    return render_template("profile.html")


@app.route("/pfizer")
def pfizer():
    data1 = mongo.db.pfizer_data.find()
    label = []
    label_percentage = []
    for x in data1:
        label.append(x["label"])
        label_percentage.append(x["count_percentage"])
    return render_template("pfizer_data.html", label=label, label_percentage=label_percentage)


@app.route("/moderna")
def moderna():
    data1 = mongo.db.moderna_data.find()
    label = []
    label_percentage = []
    for x in data1:
        label.append(x["label"])
        label_percentage.append(x["count_percentage"])
    return render_template("moderna_data.html", label=label, label_percentage=label_percentage)


@app.route("/janssen")
def janssen():
    data1 = mongo.db.janssen_data.find()
    label = []
    label_percentage = []
    for x in data1:
        label.append(x["label"])
        label_percentage.append(x["count_percentage"])
    return render_template("janssen_data.html", label=label, label_percentage=label_percentage)


@app.route("/form")
def form():
    return render_template("form.html")


@app.route("/submit", methods=["POST"])
def submit():
    loaded_rf = joblib.load("rb_boost.joblib")
    loaded_tfidf_vect = joblib.load("tfidf_vect.joblib")
    age = request.form.get("age")
    gender = request.form.get("gender")
    DOV = request.form.get("dov")
    DOR = request.form.get("dor")
    symptoms = request.form.get("Symptoms")
    medical_history = request.form.get("medical_history")
    curr_ill = request.form.get("curr_ill")
    oth_med = request.form.get("oth_med")
    died = request.form.get("died")
    hospitalized = request.form.get("hospitalized")
    ER = request.form.get("ER")
    vaccine_type = request.form.get("vaccine_type")
    vaccine_name = request.form.get("vaccine_name")
    input_text = preprocessing(symptoms)
    transformer = TfidfTransformer()
    text_tfidf = transformer.fit_transform(
        loaded_tfidf_vect.transform([input_text]))
    s = loaded_rf.predict(text_tfidf)
    mongo.db.user_data.insert_one(
        {
            "age": age,
            "gender": gender,
            "DOV": DOV,
            "DOR": DOR,
            "symptoms": symptoms,
            "medical_history": medical_history,
            "current_illness": curr_ill,
            "other_meds": oth_med,
            "died": died,
            "hospitalized": hospitalized,
            "ER": ER,
            "vaccine_type": vaccine_type,
            "vaccine_name": vaccine_name,
            "predict": int(s[0])
        }
    )
    loaded_rf = joblib.load("rb_boost.joblib")
    loaded_tfidf_vect = joblib.load("tfidf_vect.joblib")
    text = []
    text.append(symptoms)
    print(text)
    transformer = TfidfTransformer()
    text_tfidf = transformer.fit_transform(loaded_tfidf_vect.transform(text))
    def predict_fn_rf(x): return loaded_rf.predict_proba(x).astype(float)

    with open('data', 'rb') as f:
        foo = dill.load(f)
        res = foo.explain_instance(text_tfidf, predict_fn_rf, num_features=10)
    return render_template("submit.html", s=int(s[0]), vaccine_name=vaccine_name,
                           ER=ER, hospitalized=hospitalized, died=died, oth_med=oth_med,
                           curr_ill=curr_ill, medical_history=medical_history,
                           symptoms=symptoms, DOR=DOR, DOV=DOV, age=age, gender=gender, vaccine_type=vaccine_type, x=res.as_html()
                           )


if __name__ == '__main__':
    app.run(debug=True)
