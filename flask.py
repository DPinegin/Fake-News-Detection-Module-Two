from flask import Flask, redirect, url_for, render_template, request, send_file
result = 0

app = Flask(__name__)

@app.route("/")
def main():
    return send_file("popup.html")

@app.route('page2')
def page2():
    return send_file("true.html")
    # your code
    # return a response
@app.route('page3', methods=["post"])
def page3():
    return "Thanks"


if __name__ == "__main__":
    app.run(debug=True)