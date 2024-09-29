import numpy as np
import pandas as pd
#import preprocessor as p
import counselor
from tensorflow.keras.models import load_model
import joblib
from pathlib import Path
from PIL import Image
import streamlit as st
import imagify
from bokeh.plotting import figure, output_file, show
import math
from bokeh.palettes import Greens
from bokeh.transform import cumsum
from bokeh.models import LabelSet, ColumnDataSource
import pickle

flag = 1

# Load the trained model and vectorizer
with open('chatbot_model.pkl', 'rb') as model_file:
    model, vectorizer = pickle.load(model_file)

def chatbot_response(user_input):
    # Preprocess the input
    user_input = user_input.lower().replace('[^\w\s]', '')
    user_input_vector = vectorizer.transform([user_input])
    
    # Predict the response
    predicted_response = model.predict(user_input_vector)
    return predicted_response[0]

# Function to get user input text
def get_text(unique_key):
    x = st.text_input("You: ", key=unique_key)
    x = x.lower()
    xx = x[:8]
    global flag
    if xx == "find me":
        flag = 0
    input_text = [x]
    df_input = pd.DataFrame(input_text, columns=['User Input'])
    return df_input

# Streamlit application
def main():
    n=1
    

    qvals = {"Select an Option": 0, "Strongly Agree": 5, "Agree": 4, "Neutral": 3, "Disagree": 2, "Strongly Disagree": 1}
    banner = Image.open("img/21.png")
    banner = banner.resize((1064, 644))
    st.image(banner, use_column_width=True)

    # Typing text animation
    st.markdown(
    """
    <style>
    .typed-out {
        overflow: hidden;
        border-right: .15em solid orange;
        white-space: nowrap;
        letter-spacing: .15em;
        animation: typing 5s steps(60, end), blink-caret .75s step-end infinite;
    }

    @keyframes typing {
        from { width: 0 }
        to { width: 100% }
    }

    @keyframes blink-caret {
        from, to { border-color: transparent }
        50% { border-color: orange; }
    }
    </style>
    <p class="typed-out">Hi! I'm CareerFinder. Type 'find me' to choose your career.</p>
    """, unsafe_allow_html=True
    )

    # Ensure unique key is passed
    df3 = get_text(unique_key="secondary_input_text")
    
    if df3.loc[0, 'User Input'] == "":
        ans = "Hi, I'm CareerFinder. \nHow can I help you?"
    elif flag == 0:
        ans = "Sure, good luck!"
    else:
        ans = chatbot_response(df3.loc[0, 'User Input'])

    st.text_area("CareerFinder:", value=ans, height=100, max_chars=None)

    if flag == 0:
        st.title("PERSONALITY TEST:")
        kr = st.selectbox("Are you ready to take test?", ["Select an Option", "Yes", "No"])
        if kr == "Yes":
            kr1 = st.selectbox("Select level of education",
                               ["Select an Option", "Grade 10", "Grade 12", "Undergraduate"])
            #####################################  GRADE 10  ###########################################

            if(kr1=="Grade 10"):
                lis = []
                if (kr == "Yes"):
                    st.header("Question 1")
                    st.write("You find it engaging to write programs for computer applications.")
                    n = imagify.imageify(n)
                    inp = st.selectbox("",
                                       ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                        "Strongly Disagree"],
                                       key='1')
                    if ((inp != "Select an Option")):
                        lis.append(qvals[inp])
                        st.header("Question 2")
                        st.write("You easily grasp mathematical problems.")
                        n = imagify.imageify(n)
                        inp2 = st.selectbox("", ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                                 "Strongly Disagree"], key='2')

                        if (inp2 != "Select an Option"):
                            lis.append(qvals[inp2])
                            st.header("Question 3")
                            st.write("You find learning about individual chemical components intriguing.")
                            n = imagify.imageify(n)
                            inp3 = st.selectbox("", ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                                     "Strongly Disagree"], key='3')
                            if (inp3 != "Select an Option"):
                                lis.append(qvals[inp3])
                                st.header("Question 4")
                                st.write("You are curious about how plants and animals thrive.")
                                n = imagify.imageify(n)
                                inp4 = st.selectbox("",
                                                    ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                                     "Strongly Disagree"], key='4')
                                if (inp4 != "Select an Option"):
                                    lis.append(qvals[inp4])
                                    st.header("Question 5")
                                    st.write("You are fascinated by how fundamental elements of the universe interact.")
                                    n = imagify.imageify(n)
                                    inp5 = st.selectbox("",
                                                        ["Select an Option", "Strongly Agree", "Agree", "Neutral",
                                                         "Disagree",
                                                         "Strongly Disagree"], key='5')
                                    if (inp5 != "Select an Option"):
                                        lis.append(qvals[inp5])
                                        st.header("Question 6")
                                        st.write(
                                           "You excel at accounting and business management.")
                                        n = imagify.imageify(n)
                                        inp6 = st.selectbox("",
                                                            ["Select an Option", "Strongly Agree", "Agree", "Neutral",
                                                             "Disagree",
                                                             "Strongly Disagree"], key='6')
                                        if (inp6 != "Select an Option"):
                                            lis.append(qvals[inp6])
                                            st.header("Question 7")
                                            st.write(
                                               "You are interested in understanding human behavior, relationships, and thought patterns.")
                                            n = imagify.imageify(n)
                                            inp7 = st.selectbox("",
                                                                ["Select an Option", "Strongly Agree", "Agree", "Neutral",
                                                                 "Disagree",
                                                                 "Strongly Disagree"], key='7')
                                            if (inp7 != "Select an Option"):
                                                lis.append(qvals[inp7])
                                                st.header("Question 8")
                                                st.write(
                                                   "You believe it’s important to be knowledgeable about historical events.")
                                                n = imagify.imageify(n)
                                                inp8 = st.selectbox("",
                                                                    ["Select an Option", "Strongly Agree", "Agree",
                                                                     "Neutral",
                                                                     "Disagree",
                                                                     "Strongly Disagree"], key='8')
                                                if (inp8 != "Select an Option"):
                                                    lis.append(qvals[inp8])
                                                    st.header("Question 9")
                                                    st.write(
                                                        "You envision yourself as a professional athlete or trainer.")
                                                    n = imagify.imageify(n)
                                                    inp9 = st.selectbox("",
                                                                        ["Select an Option", "Strongly Agree", "Agree",
                                                                         "Neutral",
                                                                         "Disagree",
                                                                         "Strongly Disagree"], key='9')
                                                    if (inp9 != "Select an Option"):
                                                        lis.append(qvals[inp9])
                                                        st.header("Question 10")
                                                        st.write(
                                                            "You enjoy creating artistic works.")
                                                        n = imagify.imageify(n)
                                                        inp10 = st.selectbox("",
                                                                             ["Select an Option", "Strongly Agree", "Agree",
                                                                              "Neutral",
                                                                              "Disagree",
                                                                              "Strongly Disagree"], key='10')
                                                        if (inp10 != "Select an Option"):
                                                            lis.append(qvals[inp10])
                                                            st.success("Test Completed")
                                                            #st.write(lis)
                                                            st.title("RESULTS:")
                                                            df = pd.read_csv(r"Subjects.csv")

                                                            input_list = lis

                                                            subjects = {1: "Computers",
                                                                        2: "Mathematics",
                                                                        3: "Chemistry",
                                                                        4: "Biology",
                                                                        5: "Physics",
                                                                        6: "Commerce",
                                                                        7: "Psychology",
                                                                        8: "History",
                                                                        9: "Physical Education",
                                                                        10: "Design"}

                                                            def output(listofanswers):
                                                                class my_dictionary(dict):
                                                                    def __init__(self):
                                                                        self = dict()

                                                                    def add(self, key, value):
                                                                        self[key] = value

                                                                ques = my_dictionary()

                                                                for i in range(0, 10):
                                                                    ques.add(i, input_list[i])

                                                                all_scores = []

                                                                for i in range(9):
                                                                    all_scores.append(ques[i] / 5)

                                                                li = []

                                                                for i in range(len(all_scores)):
                                                                    li.append([all_scores[i], i])
                                                                li.sort(reverse=True)
                                                                sort_index = []
                                                                for x in li:
                                                                    sort_index.append(x[1] + 1)
                                                                all_scores.sort(reverse=True)

                                                                a = sort_index[0:5]
                                                                b = all_scores[0:5]
                                                                s = sum(b)
                                                                d = list(map(lambda x: x * (100 / s), b))

                                                                return a, d

                                                            l, data = output(input_list)

                                                            out = []
                                                            for i in range(0, 5):
                                                                n = l[i]
                                                                c = subjects[n]
                                                                out.append(c)

                                                            output_file("pie.html")

                                                            graph = figure(title="Recommended subjects", height=500,
                                                                           width=500)
                                                            radians = [math.radians((percent / 100) * 360) for percent
                                                                       in data]

                                                            start_angle = [math.radians(0)]
                                                            prev = start_angle[0]
                                                            for i in radians[:-1]:
                                                                start_angle.append(i + prev)
                                                                prev = i + prev

                                                            end_angle = start_angle[1:] + [math.radians(0)]

                                                            x = 0
                                                            y = 0

                                                            radius = 0.8

                                                            color = Greens[len(out)]
                                                            graph.xgrid.visible = False
                                                            graph.ygrid.visible = False
                                                            graph.xaxis.visible = False
                                                            graph.yaxis.visible = False

                                                            for i in range(len(out)):
                                                                graph.wedge(x, y, radius,
                                                                            start_angle=start_angle[i],
                                                                            end_angle=end_angle[i],
                                                                            color=color[i],
                                                                            legend_label=out[i] + "-" + str(
                                                                                round(data[i])) + "%")

                                                            graph.add_layout(graph.legend[0], 'right')
                                                            st.bokeh_chart(graph, use_container_width=True)
                                                            labels = LabelSet(x='text_pos_x', y='text_pos_y',
                                                                                text='percentage', level='glyph',
                                                                                angle=0, render_mode='canvas')
                                                            graph.add_layout(labels)

                                                            st.header('More information on the subjects')
                                                            # We'll be using a csv file for that
                                                            for i in range(0, 5):
                                                                st.subheader(subjects[int(l[i])])
                                                                st.write(df['about'][int(l[i]) - 1])

                                                            st.header('Choice of Degrees')
                                                            # We'll be using a csv file for that
                                                            for i in range(0, 5):
                                                                st.subheader(subjects[int(l[i])])
                                                                st.write(df['further career'][int(l[i]) - 1])

                                                            st.header('Trends over the years')
                                                            # We'll be using a csv file for that
                                                           

                                                            def Convert(string):
                                                                li = list(string.split(","))
                                                                li = list(map(float, li))
                                                                return li

                                                            x = ['2000', '2005', '2010', '2015', '2020']
                                                            y = []
                                                            for i in range(0, 5):
                                                                t = Convert(df['trends'][int(l[i]) - 1])
                                                                y.append(t)
                                                            output_file("line.html")
                                                            graph2 = figure(title="Trends")

                                                            graph2.line(x, y[0], line_color="Purple",
                                                                        legend_label=out[0])
                                                            graph2.line(x, y[1], line_color="Blue",
                                                                        legend_label=out[1])
                                                            graph2.line(x, y[2], line_color="Green",
                                                                        legend_label=out[2])
                                                            graph2.line(x, y[3], line_color="Magenta",
                                                                        legend_label=out[3])
                                                            graph2.line(x, y[4], line_color="Red",
                                                                        legend_label=out[4])

                                                            graph2.add_layout(graph2.legend[0], 'right')
                                                            st.bokeh_chart(graph2, use_container_width=True)
                                                            #banner1 = Image.open("img/coun.png")
                                                            #st.image(banner1, use_column_width=True)
                                                           


        ##########################################  GRADE 12  ########################################################

            elif (kr1 == "Grade 12"):
                lis = []
                st.header("Question 1")
                st.write("You enjoy discussing and negotiating issues in public settings.")
                n = imagify.imageify(n)
                inp = st.selectbox("",
                                   ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                    "Strongly Disagree"],
                                   key='1')
                if ((inp != "Select an Option")):
                    lis.append(qvals[inp])
                    st.header("Question 2")
                    st.write("You look forward to studying human anatomy and providing first aid.")
                    n = imagify.imageify(n)
                    inp2 = st.selectbox("", ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                             "Strongly Disagree"], key='2')

                    if (inp2 != "Select an Option"):
                        lis.append(qvals[inp2])
                        st.header("Question 3")
                        st.write("You are capable of leading a team and managing projects effectively.")
                        n = imagify.imageify(n)
                        inp3 = st.selectbox("", ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                                 "Strongly Disagree"], key='3')
                        if (inp3 != "Select an Option"):
                            lis.append(qvals[inp3])
                            st.header("Question 4")
                            st.write("You find working with tools, equipment, and machinery enjoyable.")
                            n = imagify.imageify(n)
                            inp4 = st.selectbox("",
                                                ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                                 "Strongly Disagree"], key='4')
                            if (inp4 != "Select an Option"):
                                lis.append(qvals[inp4])
                                st.header("Question 5")
                                st.write(
                                    "You handle budgeting, costing, and estimating for a business with ease.")
                                n = imagify.imageify(n)
                                inp5 = st.selectbox("",
                                                    ["Select an Option", "Strongly Agree", "Agree", "Neutral",
                                                     "Disagree",
                                                     "Strongly Disagree"], key='5')
                                if (inp5 != "Select an Option"):
                                    lis.append(qvals[inp5])
                                    st.header("Question 6")
                                    st.write(
                                        "You envision participating in competitive sports with the goal of becoming a professional.")
                                    n = imagify.imageify(n)
                                    inp6 = st.selectbox("",
                                                        ["Select an Option", "Strongly Agree", "Agree", "Neutral",
                                                         "Disagree",
                                                         "Strongly Disagree"], key='6')
                                    if (inp6 != "Select an Option"):
                                        lis.append(qvals[inp6])
                                        st.header("Question 7")
                                        st.write(
                                           "You don’t get fatigued by tasks like translation, reading, and language correction.")
                                        n = imagify.imageify(n)
                                        inp7 = st.selectbox("",
                                                            ["Select an Option", "Strongly Agree", "Agree", "Neutral",
                                                             "Disagree",
                                                             "Strongly Disagree"], key='7')
                                        if (inp7 != "Select an Option"):
                                            lis.append(qvals[inp7])
                                            st.header("Question 8")
                                            st.write(
                                                "You would be excited to act in or direct a play or film.")
                                            n = imagify.imageify(n)
                                            inp8 = st.selectbox("",
                                                                ["Select an Option", "Strongly Agree", "Agree",
                                                                 "Neutral",
                                                                 "Disagree",
                                                                 "Strongly Disagree"], key='8')
                                            if (inp8 != "Select an Option"):
                                                lis.append(qvals[inp8])
                                                st.header("Question 9")
                                                st.write(
                                                   "You view sketching people or landscapes as a hobby that could turn into a career.")
                                                n = imagify.imageify(n)
                                                inp9 = st.selectbox("",
                                                                    ["Select an Option", "Strongly Agree", "Agree",
                                                                     "Neutral",
                                                                     "Disagree",
                                                                     "Strongly Disagree"], key='9')
                                                if (inp9 != "Select an Option"):
                                                    lis.append(qvals[inp9])
                                                    st.header("Question 10")
                                                    st.write(
                                                        "You are comfortable working with numbers and performing calculations most of the time.")
                                                    n = imagify.imageify(n)
                                                    inp10 = st.selectbox("",
                                                                         ["Select an Option", "Strongly Agree", "Agree",
                                                                          "Neutral",
                                                                          "Disagree",
                                                                          "Strongly Disagree"], key='10')
                                                    if (inp10 != "Select an Option"):
                                                        lis.append(qvals[inp10])
                                                        st.header("Question 11")
                                                        st.write(
                                                            "You enjoy handling clerical tasks such as filing, stock counting, and issuing receipts.")
                                                        n = imagify.imageify(n)
                                                        inp11 = st.selectbox("",
                                                                             ["Select an Option", "Strongly Agree",
                                                                              "Agree",
                                                                              "Neutral",
                                                                              "Disagree",
                                                                              "Strongly Disagree"], key='11')
                                                        if (inp11 != "Select an Option"):
                                                            lis.append(qvals[inp11])
                                                            st.header("Question 12")
                                                            st.write(
                                                               "You have a passion for learning about the culture and lifestyle of different societies.")
                                                            n = imagify.imageify(n)
                                                            inp12 = st.selectbox("",
                                                                                 ["Select an Option", "Strongly Agree",
                                                                                  "Agree",
                                                                                  "Neutral",
                                                                                  "Disagree",
                                                                                  "Strongly Disagree"], key='12')
                                                            if (inp12 != "Select an Option"):
                                                                lis.append(qvals[inp12])
                                                                st.header("Question 13")
                                                                st.write(
                                                                    "You see yourself teaching children and young people on a daily basis.")
                                                                n = imagify.imageify(n)
                                                                inp13 = st.selectbox("",
                                                                                     ["Select an Option",
                                                                                      "Strongly Agree", "Agree",
                                                                                      "Neutral",
                                                                                      "Disagree",
                                                                                      "Strongly Disagree"], key='13')
                                                                if (inp13 != "Select an Option"):
                                                                    lis.append(qvals[inp13])
                                                                    st.header("Question 14")
                                                                    st.write(
                                                                       "You are confident in your ability to persevere in roles within the army or police force.")
                                                                    n = imagify.imageify(n)
                                                                    inp14 = st.selectbox("",
                                                                                         ["Select an Option",
                                                                                          "Strongly Agree", "Agree",
                                                                                          "Neutral",
                                                                                          "Disagree",
                                                                                          "Strongly Disagree"],
                                                                                         key='14')
                                                                    if (inp14 != "Select an Option"):
                                                                        lis.append(qvals[inp14])
                                                                        st.header("Question 15")
                                                                        st.write(
                                                                           "You excel at introducing new products to customers and persuading them to make a purchase.")
                                                                        n = imagify.imageify(n)
                                                                        inp15 = st.selectbox("",
                                                                                             ["Select an Option",
                                                                                              "Strongly Agree", "Agree",
                                                                                              "Neutral",
                                                                                              "Disagree",
                                                                                              "Strongly Disagree"],
                                                                                             key='15')
                                                                        if (inp15 != "Select an Option"):
                                                                            lis.append(qvals[inp10])
                                                                            st.success("Test Completed")
                                                                            #st.write(lis)
                                                                            st.title("RESULTS:")
                                                                            df = pd.read_csv(r"Graduate.csv")

                                                                            input_list = lis

                                                                            streams = {1: "Law",
                                                                                       2: "Healthcare",
                                                                                       3: "Management",
                                                                                       4: "Engineering",
                                                                                       5: "Finance",
                                                                                       6: "Sports",
                                                                                       7: "Language and communication",
                                                                                       8: "Performing Arts",
                                                                                       9: "Applied and Visual arts",
                                                                                       10: "Science and math",
                                                                                       11: "Clerical and secretarial",
                                                                                       12: "Social Science",
                                                                                       13: "Education and Social Support",
                                                                                       14: "Armed Forces",
                                                                                       15: "Marketing and sales"}

                                                                            def output(listofanswers):
                                                                                class my_dictionary(dict):
                                                                                    def __init__(self):
                                                                                        self = dict()

                                                                                    def add(self, key, value):
                                                                                        self[key] = value

                                                                                ques = my_dictionary()

                                                                                for i in range(0, 15):
                                                                                    ques.add(i, input_list[i])

                                                                                all_scores = []

                                                                                for i in range(14):
                                                                                    all_scores.append(ques[i] / 5)

                                                                                li = []

                                                                                for i in range(len(all_scores)):
                                                                                    li.append([all_scores[i], i])
                                                                                li.sort(reverse=True)
                                                                                sort_index = []
                                                                                for x in li:
                                                                                    sort_index.append(x[1] + 1)
                                                                                all_scores.sort(reverse=True)

                                                                                a = sort_index[0:5]
                                                                                b = all_scores[0:5]
                                                                                s = sum(b)
                                                                                d = list(
                                                                                    map(lambda x: x * (100 / s), b))

                                                                                return a, d

                                                                            l, data = output(input_list)

                                                                            out = []
                                                                            for i in range(0, 5):
                                                                                n = l[i]
                                                                                c = streams[n]
                                                                                out.append(c)

                                                                            output_file("pie.html")

                                                                            graph = figure(title="Recommended fields",
                                                                                           height=500, width=500)
                                                                            radians = [
                                                                                math.radians((percent / 100) * 360) for
                                                                                percent in data]

                                                                            start_angle = [math.radians(0)]
                                                                            prev = start_angle[0]
                                                                            for i in radians[:-1]:
                                                                                start_angle.append(i + prev)
                                                                                prev = i + prev

                                                                            end_angle = start_angle[1:] + [
                                                                                math.radians(0)]

                                                                            x = 0
                                                                            y = 0

                                                                            radius = 0.8

                                                                            color = Greens[len(out)]
                                                                            graph.xgrid.visible = False
                                                                            graph.ygrid.visible = False
                                                                            graph.xaxis.visible = False
                                                                            graph.yaxis.visible = False

                                                                            for i in range(len(out)):
                                                                                graph.wedge(x, y, radius,
                                                                                            start_angle=start_angle[i],
                                                                                            end_angle=end_angle[i],
                                                                                            color=color[i],
                                                                                            legend_label=out[
                                                                                                             i] + "-" + str(
                                                                                                round(data[i])) + "%")

                                                                            graph.add_layout(graph.legend[0],
                                                                                                'right')
                                                                            st.bokeh_chart(graph,
                                                                                            use_container_width=True)
                                                                            labels = LabelSet(x='text_pos_x',
                                                                                                y='text_pos_y',
                                                                                                text='percentage',
                                                                                                level='glyph',
                                                                                                angle=0,
                                                                                                render_mode='canvas')
                                                                            graph.add_layout(labels)

                                                                            st.header(
                                                                                'More information on the fields')
                                                                            # We'll be using a csv file for that
                                                                            for i in range(0, 5):
                                                                                st.subheader(streams[int(l[i])])
                                                                                st.write(df['About'][int(l[i]) - 1])

                                                                            st.header('Average annual salary')
                                                                            # We'll be using a csv file for that
                                                                            for i in range(0, 5):
                                                                                st.subheader(streams[int(l[i])])
                                                                                st.write("Rs. "+ str(
                                                                                    df['avgsal'][int(l[i]) - 1]))

                                                                            st.header('Trends over the years')
                                                                            # We'll be using a csv file for that
                                                                            

                                                                            def Convert(string):
                                                                                li = list(string.split(","))
                                                                                li = list(map(float, li))
                                                                                return li

                                                                            x = ['2000', '2005', '2010', '2015', '2020']
                                                                            y = []
                                                                            for i in range(0, 5):
                                                                                t = Convert(df['trends'][int(l[i]) - 1])
                                                                                y.append(t)
                                                                            output_file("line.html")
                                                                            graph2 = figure(title="Trends")

                                                                            graph2.line(x, y[0], line_color="Purple",
                                                                                        legend_label=out[0])
                                                                            graph2.line(x, y[1], line_color="Blue",
                                                                                        legend_label=out[1])
                                                                            graph2.line(x, y[2], line_color="Green",
                                                                                        legend_label=out[2])
                                                                            graph2.line(x, y[3], line_color="Magenta",
                                                                                        legend_label=out[3])
                                                                            graph2.line(x, y[4], line_color="Red",
                                                                                        legend_label=out[4])

                                                                            graph2.add_layout(graph2.legend[0], 'right')
                                                                            st.bokeh_chart(graph2,
                                                                                           use_container_width=True)

                                                                            #banner1 = Image.open("img/coun.png")
                                                                            #st.image(banner1, use_column_width=True)
                                                                            st.header(
                                                                                "Contacts of experts from various fields")


                                                                            for i in range(0, 5):
                                                                                st.subheader(streams[int(l[i])])
                                                                                xl = (
                                                                                df['contacts'][int(l[i]) - 1]).split(
                                                                                    ",")
                                                                                for k in xl:
                                                                                    ml = list(k.split(","))
                                                                                    for kk in ml:
                                                                                        st.write(kk, sep="\n")




            ######################################  UNDERGRADUATE ##########################################

            elif (kr1 == "Undergraduate"):
                lis = []
                if (kr == "Yes"):
                    st.header("Question 1")
                    st.write("You can manage all aspects of information security and ensure the protection of a company's digital data.")
                    #n = imagify.imageify(n)
                    inp = st.selectbox("",
                                       ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                        "Strongly Disagree"],
                                       key='1')
                    if ((inp != "Select an Option")):
                        lis.append(qvals[inp])
                        st.header("Question 2")
                        st.write("You enjoy analyzing a company's business and information needs and using that data to create processes that help meet strategic goals.")
                        #n = imagify.imageify(n)
                        inp2 = st.selectbox("", ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                                 "Strongly Disagree"], key='2')

                        if (inp2 != "Select an Option"):
                            lis.append(qvals[inp2])
                            st.header("Question 3")
                            st.write("You are skilled at identifying problems and either designing new systems or improving existing ones to enhance their efficiency.")
                            #n = imagify.imageify(n)
                            inp3 = st.selectbox("", ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                                     "Strongly Disagree"], key='3')
                            if (inp3 != "Select an Option"):
                                lis.append(qvals[inp3])
                                st.header("Question 4")
                                st.write("You excel at working with databases and large datasets, whether it's designing, developing, modifying, or editing them.")
                                #n = imagify.imageify(n)
                                inp4 = st.selectbox("",
                                                    ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree",
                                                     "Strongly Disagree"], key='4')
                                if (inp4 != "Select an Option"):
                                    lis.append(qvals[inp4])
                                    st.header("Question 5")
                                    st.write(
                                       "You are adept at using BI software tools to analyze, compare, visualize, and clearly present data.")
                                    #n = imagify.imageify(n)
                                    inp5 = st.selectbox("",
                                                        ["Select an Option", "Strongly Agree", "Agree", "Neutral",
                                                         "Disagree",
                                                         "Strongly Disagree"], key='5')
                                    if (inp5 != "Select an Option"):
                                        lis.append(qvals[inp5])
                                        st.header("Question 6")
                                        st.write(
                                            "You are skilled in setting up and supporting Microsoft's Dynamics CRM system.")
                                        #n = imagify.imageify(n)
                                        inp6 = st.selectbox("",
                                                            ["Select an Option", "Strongly Agree", "Agree", "Neutral",
                                                             "Disagree",
                                                             "Strongly Disagree"], key='6')
                                        if (inp6 != "Select an Option"):
                                            lis.append(qvals[inp6])
                                            st.header("Question 7")
                                            st.write(
                                                "You can bring creativity and innovation to developing user-friendly mobile applications.")
                                            #n = imagify.imageify(n)
                                            inp7 = st.selectbox("",
                                                                ["Select an Option", "Strongly Agree", "Agree", "Neutral",
                                                                 "Disagree",
                                                                 "Strongly Disagree"], key='7')
                                            if (inp7 != "Select an Option"):
                                                lis.append(qvals[inp7])
                                                st.header("Question 8")
                                                st.write(
                                                    "You are effective in roles that blend psychology, business, market research, design, and technology.")
                                                #n = imagify.imageify(n)
                                                inp8 = st.selectbox("",
                                                                    ["Select an Option", "Strongly Agree", "Agree",
                                                                     "Neutral",
                                                                     "Disagree",
                                                                     "Strongly Disagree"], key='8')
                                                if (inp8 != "Select an Option"):
                                                    lis.append(qvals[inp8])
                                                    st.header("Question 9")
                                                    st.write(
                                                       "You are reliable in maintaining quality systems like laboratory controls, document management, and training to ensure smooth manufacturing processes.")
                                                    #n = imagify.imageify(n)
                                                    inp9 = st.selectbox("",
                                                                        ["Select an Option", "Strongly Agree", "Agree",
                                                                         "Neutral",
                                                                         "Disagree",
                                                                         "Strongly Disagree"], key='9')
                                                    if (inp9 != "Select an Option"):
                                                        lis.append(qvals[inp9])
                                                        st.header("Question 10")
                                                        st.write(
                                                            "You are passionate about designing and developing websites, whether it's the front-end or back-end.")
                                                        #n = imagify.imageify(n)
                                                        inp10 = st.selectbox("",
                                                                             ["Select an Option", "Strongly Agree", "Agree",
                                                                              "Neutral",
                                                                              "Disagree",
                                                                              "Strongly Disagree"], key='10')
                                                        if (inp10 != "Select an Option"):
                                                            lis.append(qvals[inp10])
                                                            st.success("Test Completed")
                                                            #st.write(lis)

                                                            st.title("RESULTS:")
                                                            df = pd.read_csv(r'Occupations.csv', encoding= 'windows-1252')

                                                            input_list = lis

                                                            professions = {1: "Systems Security Administrator",
                                                                        2: "Business Systems Analyst",
                                                                        3: "Software Systems Engineer",
                                                                        4: "Database Developer",
                                                                        5: "Business Intelligence Analyst",
                                                                        6: "CRM Technical Developer",
                                                                        7: "Mobile Applications Developer",
                                                                        8: "UX Designer",
                                                                        9: "Quality Assurance Associate",
                                                                        10: "Web Developer"}

                                                            def output(listofanswers):
                                                                class my_dictionary(dict):
                                                                    def __init__(self):
                                                                        self = dict()

                                                                    def add(self, key, value):
                                                                        self[key] = value

                                                                ques = my_dictionary()

                                                                for i in range(0, 10):
                                                                    ques.add(i, input_list[i])

                                                                all_scores = []

                                                                for i in range(9):
                                                                    all_scores.append(ques[i] / 5)

                                                                li = []

                                                                for i in range(len(all_scores)):
                                                                    li.append([all_scores[i], i])
                                                                li.sort(reverse=True)
                                                                sort_index = []
                                                                for x in li:
                                                                    sort_index.append(x[1] + 1)
                                                                all_scores.sort(reverse=True)

                                                                a = sort_index[0:5]
                                                                b = all_scores[0:5]
                                                                s = sum(b)
                                                                d = list(map(lambda x: x * (100 / s), b))

                                                                return a, d

                                                            l, data = output(input_list)

                                                            out = []
                                                            for i in range(0, 5):
                                                                n = l[i]
                                                                c = professions[n]
                                                                out.append(c)

                                                            output_file("pie.html")

                                                            graph = figure(title="Recommended professions", height=500,
                                                                           width=500)
                                                            radians = [math.radians((percent / 100) * 360) for percent
                                                                       in data]

                                                            start_angle = [math.radians(0)]
                                                            prev = start_angle[0]
                                                            for i in radians[:-1]:
                                                                start_angle.append(i + prev)
                                                                prev = i + prev

                                                            end_angle = start_angle[1:] + [math.radians(0)]

                                                            x = 0
                                                            y = 0

                                                            radius = 0.8

                                                            color = Greens[len(out)]
                                                            graph.xgrid.visible = False
                                                            graph.ygrid.visible = False
                                                            graph.xaxis.visible = False
                                                            graph.yaxis.visible = False

                                                            for i in range(len(out)):
                                                                graph.wedge(x, y, radius,
                                                                            start_angle=start_angle[i],
                                                                            end_angle=end_angle[i],
                                                                            color=color[i],
                                                                            legend_label=out[i] + "-" + str(
                                                                                round(data[i])) + "%")

                                                            graph.add_layout(graph.legend[0], 'right')
                                                            st.bokeh_chart(graph, use_container_width=True)
                                                            labels = LabelSet(x='text_pos_x', y='text_pos_y',
                                                                                text='percentage', level='glyph',
                                                                                angle=0, render_mode='canvas')
                                                            graph.add_layout(labels)
                                                            st.header('More information on the professions')
                                                            # We'll be using a csv file for that
                                                            for i in range(0, 5):
                                                                st.subheader(professions[int(l[i])])
                                                                st.write(df['Information'][int(l[i]) - 1])

                                                            st.header('Monthly Income')
                                                            # We'll be using a csv file for that
                                                            for i in range(0, 5):
                                                                st.subheader(professions[int(l[i])])
                                                                st.write("Rs. " + str(df['Income'][int(l[i]) - 1]))

                                                            st.header('Trends over the years')
                                                            # We'll be using a csv file for that
                                                        

                                                            def Convert(string):
                                                                li = list(string.split(","))
                                                                li = list(map(float, li))
                                                                return li

                                                            x = ['2000', '2005', '2010', '2015', '2020']
                                                            y = []
                                                            for i in range(0, 5):
                                                                t = Convert(df['trends'][int(l[i]) - 1])
                                                                y.append(t)
                                                            output_file("line.html")
                                                            graph2 = figure(title="Trends")

                                                            graph2.line(x, y[0], line_color="Purple",
                                                                        legend_label=out[0])
                                                            graph2.line(x, y[1], line_color="Blue", legend_label=out[1])
                                                            graph2.line(x, y[2], line_color="Green",
                                                                        legend_label=out[2])
                                                            graph2.line(x, y[3], line_color="Magenta",
                                                                        legend_label=out[3])
                                                            graph2.line(x, y[4], line_color="Red", legend_label=out[4])

                                                            graph2.add_layout(graph2.legend[0], 'right')
                                                            st.bokeh_chart(graph2, use_container_width=True)
                                                            #banner1 = Image.open("img/coun.png")
                                                            #st.image(banner1, use_column_width=True)
                                                            st.header("Contacts of experts from various fields")
                                                            for i in range(0, 5):
                                                                st.subheader(professions[int(l[i])])
                                                                xl=(df['contacts'][int(l[i]) - 1]).split(",")
                                                                for k in xl:
                                                                    ml=list(k.split(","))
                                                                    for kk in ml:
                                                                        st.write(kk,sep="\n")



if __name__=="__main__":
    main()
