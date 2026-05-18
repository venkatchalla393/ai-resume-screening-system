from flask import Flask, render_template, request
import PyPDF2

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():

    file = request.files["resume"]

    # check file selected
    if file.filename == "":
        return "No file selected"

    # allow only pdf
    if not file.filename.endswith(".pdf"):
        return "Please upload PDF file only"

    try:

        # read pdf
        pdf_reader = PyPDF2.PdfReader(file)

        # extract text
        resume_text = ""

        for page in pdf_reader.pages:
            resume_text += page.extract_text()

        # skills list
        skills = [
            "python",
            "sql",
            "flask",
            "machine learning",
            "pandas",
            "numpy",
            "tensorflow",
            "keras",
            "scikit-learn",
            "data analysis"
        ]

        # detected skills
        detected_skills = []

        for skill in skills:

            if skill.lower() in resume_text.lower():
                detected_skills.append(skill)

        # get job description
        job_description = request.form["job_description"]

        # convert text into vectors
        text = [resume_text, job_description]

        tfidf = TfidfVectorizer()

        vectors = tfidf.fit_transform(text)

        # similarity
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])

        # percentage
        match_percentage = round(similarity[0][0] * 100, 2)

        # result ui
        return f"""

<!DOCTYPE html>
<html lang='en'>

<head>

    <meta charset='UTF-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1.0'>

    <title>Resume Analysis</title>

    <link
    href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css'
    rel='stylesheet'>

    <link
    rel='stylesheet'
    href='https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css'>

    <style>

        body{{
            background: linear-gradient(135deg, #0f172a, #1e293b, #2563eb);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 40px;
            font-family: Arial;
        }}

        .result-card{{
            width: 850px;
            background: rgba(255,255,255,0.12);
            backdrop-filter: blur(15px);
            border-radius: 25px;
            padding: 40px;
            color: white;
            box-shadow: 0px 10px 40px rgba(0,0,0,0.4);
            animation: fadeIn 1s ease;
        }}

        @keyframes fadeIn {{
            from{{
                opacity: 0;
                transform: translateY(30px);
            }}

            to{{
                opacity: 1;
                transform: translateY(0px);
            }}
        }}

        .score{{
            font-size: 70px;
            font-weight: bold;
            color: #38bdf8;
        }}

        .skill-badge{{
            background-color: #2563eb;
            padding: 10px 18px;
            border-radius: 30px;
            margin: 5px;
            display: inline-block;
            font-weight: 500;
        }}

        .section{{
            background: rgba(255,255,255,0.08);
            padding: 20px;
            border-radius: 15px;
            margin-top: 25px;
        }}

        .btn-home{{
            margin-top: 25px;
            background: linear-gradient(to right, #06b6d4, #2563eb);
            border: none;
            padding: 12px 25px;
            border-radius: 10px;
            color: white;
            font-weight: bold;
            text-decoration: none;
            display: inline-block;
        }}

        .btn-home:hover{{
            opacity: 0.9;
            color: white;
        }}

        .resume-text{{
            max-height: 250px;
            overflow-y: auto;
            line-height: 1.7;
        }}

    </style>

</head>

<body>

    <div class='result-card'>

        <h1 class='text-center mb-4'>
            <i class='bi bi-robot'></i>
            Resume Analysis Result
        </h1>

        <div class='text-center'>

            <div class='score'>
                {match_percentage}%
            </div>

            <h4>Resume Match Score</h4>

        </div>

        <div class='section'>

            <h3>
                <i class='bi bi-stars'></i>
                Detected Skills
            </h3>

            {''.join([f"<span class='skill-badge'>{skill}</span>" for skill in detected_skills])}

        </div>

        <div class='section'>

            <h3>
                <i class='bi bi-file-earmark-text'></i>
                Extracted Resume Text
            </h3>

            <div class='resume-text'>
                <p>{resume_text}</p>
            </div>

        </div>

        <div class='text-center'>

            <a href='/' class='btn-home'>
                <i class='bi bi-arrow-repeat'></i>
                Analyze Another Resume
            </a>

        </div>

    </div>

</body>

</html>

"""

    except:
        return "Error reading PDF file. Please upload valid PDF."
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=50000, debug=True)