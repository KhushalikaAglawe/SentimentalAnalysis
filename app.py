from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

documents = [
    {
        "title": "Indian Classical Music",
        "content": "Indian classical music includes Hindustani and Carnatic music with ragas talas rhythm and traditional instruments."
    },
    {
        "title": "Western Music",
        "content": "Western music includes classical jazz rock pop blues and electronic music with different melodies rhythms and instruments."
    },
    {
        "title": "Bollywood Music",
        "content": "Bollywood music includes Hindi film songs romantic songs dance songs playback singing and musical compositions."
    },
    {
        "title": "Pop Music",
        "content": "Pop music is popular music with catchy melodies simple lyrics strong rhythms and modern production."
    },
    {
        "title": "Rock Music",
        "content": "Rock music commonly uses electric guitars bass guitar drums strong rhythms and powerful vocals."
    },
    {
        "title": "Music Production",
        "content": "Music production involves recording vocals instruments mixing mastering sound effects and audio editing."
    },

    {
        "title": "Jazz Music",
        "content": "Jazz music features improvisation swing rhythms saxophones trumpets piano bass drums and expressive melodies."
    },
    {
        "title": "Classical Piano",
        "content": "Classical piano music includes compositions by famous composers with complex melodies harmony rhythm and musical techniques."
    },
    {
        "title": "Electronic Music",
        "content": "Electronic music uses synthesizers drum machines computers samples digital sounds and electronic beats."
    },
    {
        "title": "Hip Hop Music",
        "content": "Hip hop music includes rap vocals beats rhythm sampling DJ techniques and strong lyrical expression."
    },
    {
        "title": "Rap Music",
        "content": "Rap music focuses on rhythmic vocals rhyming lyrics beats storytelling and powerful verbal expression."
    },
    {
        "title": "Blues Music",
        "content": "Blues music is known for emotional vocals guitar melodies repeated patterns and expressive musical performances."
    },
    {
        "title": "Folk Music",
        "content": "Folk music represents traditional culture using regional instruments traditional melodies songs and storytelling."
    },
    {
        "title": "Guitar Music",
        "content": "Guitar music uses acoustic or electric guitars with chords melodies riffs solos and different playing techniques."
    },
    {
        "title": "Drum Music",
        "content": "Drums create rhythm and beats in music and are important in rock pop jazz electronic and traditional music."
    },
    {
        "title": "Indian Folk Music",
        "content": "Indian folk music includes regional songs traditional instruments cultural celebrations festivals and local storytelling."
    },
    {
        "title": "Carnatic Music",
        "content": "Carnatic music is a traditional South Indian classical music system based on ragas talas compositions and vocal performance."
    },
    {
        "title": "Hindustani Music",
        "content": "Hindustani music is North Indian classical music based on ragas improvisation rhythm talas and traditional performances."
    },
    {
        "title": "Devotional Music",
        "content": "Devotional music includes bhajans chants spiritual songs religious lyrics traditional instruments and peaceful melodies."
    },
    {
        "title": "Romantic Songs",
        "content": "Romantic songs focus on love relationships emotions feelings memories and soft musical melodies."
    },
    {
        "title": "Dance Music",
        "content": "Dance music uses energetic beats rhythms electronic sounds catchy melodies and fast musical patterns."
    },
    {
        "title": "Instrumental Music",
        "content": "Instrumental music focuses on musical instruments without vocals and includes melodies harmony rhythm and musical arrangements."
    },
    {
        "title": "Acoustic Music",
        "content": "Acoustic music uses natural instruments such as acoustic guitar piano violin drums and vocals without heavy electronic effects."
    },
    {
        "title": "Music Theory",
        "content": "Music theory explains notes scales chords harmony melody rhythm intervals keys and musical composition."
    },
    {
        "title": "Melody",
        "content": "Melody is a sequence of musical notes that creates a recognizable musical idea or tune."
    },
    {
        "title": "Harmony",
        "content": "Harmony combines different musical notes and chords to support melodies and create musical depth."
    },
    {
        "title": "Rhythm",
        "content": "Rhythm organizes beats and musical sounds over time and creates patterns used in different types of music."
    },
    {
        "title": "Musical Instruments",
        "content": "Musical instruments produce sound and include guitars pianos violins drums flutes keyboards and traditional instruments."
    },
    {
        "title": "Singing",
        "content": "Singing uses the human voice to produce melodies lyrics harmonies and different vocal expressions."
    },
    {
        "title": "Music Recording",
        "content": "Music recording captures vocals and instruments using microphones audio interfaces recording software and studio equipment."
    },
    {
        "title": "Audio Mixing",
        "content": "Audio mixing combines vocals instruments effects volume levels equalization compression and stereo positioning."
    },
    {
        "title": "Music Mastering",
        "content": "Music mastering is the final audio process that improves loudness balance clarity and consistency of a music track."
    },
    {
        "title": "Music Streaming",
        "content": "Music streaming allows listeners to access songs albums playlists artists and podcasts through online platforms."
    },
    {
        "title": "Live Music",
        "content": "Live music includes concerts performances stages musicians singers audiences lighting sound systems and live instruments."
    },
    {
        "title": "Music Festivals",
        "content": "Music festivals bring artists audiences live performances multiple stages concerts different genres and musical celebrations."
    },
    {
        "title": "Film Music",
        "content": "Film music uses background scores songs themes instruments and soundtracks to support emotions and scenes in movies."
    }
]
texts = [doc["content"] for doc in documents]

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(texts)


@app.route("/", methods=["GET", "POST"])
def home():

    results = []
    query = ""

    if request.method == "POST":

        query = request.form["query"]

        query_vector = vectorizer.transform([query])

        scores = cosine_similarity(
            query_vector,
            tfidf_matrix
        )[0]

        for i, score in enumerate(scores):

            if score > 0:
                results.append({
                    "title": documents[i]["title"],
                    "content": documents[i]["content"],
                    "score": round(score, 4)
                })

        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

    return render_template(
        "index.html",
        results=results,
        query=query
    )


if __name__ == "__main__":
    app.run(debug=True)