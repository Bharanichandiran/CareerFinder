import numpy as np
import pandas as pd
#import preprocessor as p
#import counselor
#from tensorflow.keras.models import load_model
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
    <h5 class="typed-out">Hi! I'm CareerFinder. Type 'find me' to choose your career.</h5>
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
            kr1 = st.selectbox("Select level of education you completed",
                               ["Select an Option", "Grade 10", "Grade 12", "Undergraduate"])
            #####################################  GRADE 10  ###########################################

            if kr1 == "Grade 10":
                lis = []
                questions = [
        "You find it engaging to write programs for computer applications.",
        "You easily grasp mathematical problems.",
        "You find learning about individual chemical components intriguing.",
        "You are curious about how plants and animals thrive.",
        "You are fascinated by how fundamental elements of the universe interact.",
        "You excel at accounting and business management.",
        "You are interested in understanding human behavior, relationships, and thought patterns.",
        "You believe it’s important to be knowledgeable about historical events.",
        "You envision yourself as a professional athlete or trainer.",
        "You enjoy creating artistic works."]
                with st.form("grade_10_form"):
                     for i, question in enumerate(questions):
                         st.header(f"Question {i+1}")
                         st.write(question)
                         inp = st.selectbox( "", 
                    ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree", "Strongly Disagree"], 
                    key=str(i+1))
                         if inp != "Select an Option":
                             lis.append(qvals[inp])
                     submit_button = st.form_submit_button(label='Submit')
                if submit_button:
                   if len(lis) == 10:
                       st.success("Test Completed")
                       st.title("RESULTS:")
                       df = pd.read_csv(r"Subjects.csv")
                       input_list = lis
                       subjects = {
                                1: "Computers",
            2: "Mathematics",
            3: "Chemistry",
            4: "Biology",
            5: "Physics",
            6: "Commerce",
            7: "Psychology",
            8: "History",
            9: "Physical Education",
            10: "Design"
                                      }
                       def output(listofanswers):
                            all_scores = [val / 5 for val in listofanswers]
                            sorted_scores = sorted(enumerate(all_scores, 1), key=lambda x: x[1], reverse=True)
                            top_indices = [x[0] for x in sorted_scores[:5]]
                            top_scores = [x[1] for x in sorted_scores[:5]]
                            normalized_scores = [x * 100 / sum(top_scores) for x in top_scores]
                            return top_indices, normalized_scores
                       l, data = output(input_list)
                       out = [subjects[i] for i in l]
                       output_file("pie.html")
                       graph = figure(title="Recommended subjects", height=500, width=500)
                       radians = [math.radians((percent / 100) * 360) for percent in data]
                       start_angle = [math.radians(0)]
                       for rad in radians[:-1]:
                           start_angle.append(start_angle[-1] + rad)
                       end_angle = start_angle[1:] + [math.radians(0)]
                       colors = Greens[len(out)]
                       for i in range(len(out)):
                            graph.wedge(
        x=0, y=0, radius=0.8,
        start_angle=start_angle[i],
        end_angle=end_angle[i],
        color=colors[i],
        legend_label=f"{out[i]} - {round(data[i])}%"
    )

                       graph.add_layout(graph.legend[0], 'right')
                       st.bokeh_chart(graph, use_container_width=True)
                       st.header('More information on the subjects')
                       for index in l:
                           st.subheader(subjects[index])
                           st.write(df['about'][index - 1])
                       st.header('Choice of Degrees')
                       for index in l:
                           st.subheader(subjects[index])
                           st.write(df['further career'][index - 1])
                       def Convert(string):
                            li = list(string.split(","))
                            li = list(map(float, li))
                            return li
                       st.header('Trends over the years')
                       l = [1, 2, 3, 4, 5]
                       x = ['2000', '2005', '2010', '2015', '2020']
                       y = []
                       for i in range(0, 5):
                            t = Convert(df['trends'][int(l[i]) - 1])
                            y.append(t)
                       output_file("line.html")
                       graph2 = figure(title="Trends", x_axis_label='Year', y_axis_label='Value')

                       # Plot each line
                       colors = ["Purple", "Blue", "Green", "Magenta", "Red"]
                       for i in range(5):
                           graph2.line(x, y[i], line_color=colors[i], legend_label=out[i])

                       graph2.add_layout(graph2.legend[0], 'right')

                           # Show the plot in Streamlit
                       st.bokeh_chart(graph2, use_container_width=True)


                                                            


        ##########################################  GRADE 12  ########################################################

            elif kr1 == "Grade 12":
                lis = []
                questions = [
       "You enjoy discussing and negotiating issues in public settings.",
        "You look forward to studying human anatomy and providing first aid.",
        "You are capable of leading a team and managing projects effectively.",
        "You find working with tools, equipment, and machinery enjoyable.",
        "You handle budgeting, costing, and estimating for a business with ease.",
        "You envision participating in competitive sports with the goal of becoming a professional.",
        "You don’t get fatigued by tasks like translation, reading, and language correction.",
        "You would be excited to act in or direct a play or film.",
        "You view sketching people or landscapes as a hobby that could turn into a career.",
        "You are comfortable working with numbers and performing calculations most of the time.",
        "You enjoy handling clerical tasks such as filing, stock counting, and issuing receipts.",
        "You have a passion for learning about the culture and lifestyle of different societies.",
        "You see yourself teaching children and young people on a daily basis.",
        "You are confident in your ability to persevere in roles within the army or police force.",
        "You excel at introducing new products to customers and persuading them to make a purchase."]
                with st.form("grade_12_form"):
                     for i, question in enumerate(questions):
                         st.header(f"Question {i+1}")
                         st.write(question)
                         inp = st.selectbox( "", 
                    ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree", "Strongly Disagree"], 
                    key=str(i+1))
                         if inp != "Select an Option":
                             lis.append(qvals[inp])
                     submit_button = st.form_submit_button(label='Submit')
                if submit_button:
                   if len(lis) == 15:
                       st.success("Test Completed")
                       st.title("RESULTS:")
                       df = pd.read_csv(r"Graduate.csv")
                       input_list = lis
                       Graduate= {
                                 1: "Law",
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
                15: "Marketing and sales"
                                      }
                       def output(listofanswers):
                            all_scores = [val / 5 for val in listofanswers]
                            sorted_scores = sorted(enumerate(all_scores, 1), key=lambda x: x[1], reverse=True)
                            top_indices = [x[0] for x in sorted_scores[:5]]
                            top_scores = [x[1] for x in sorted_scores[:5]]
                            normalized_scores = [x * 100 / sum(top_scores) for x in top_scores]
                            return top_indices, normalized_scores
                       l, data = output(input_list)
                       out = [Graduate[i] for i in l]
                       output_file("pie.html")
                       graph = figure(title="Recommended subjects", height=500, width=500)
                       radians = [math.radians((percent / 100) * 360) for percent in data]
                       start_angle = [math.radians(0)]
                       for rad in radians[:-1]:
                           start_angle.append(start_angle[-1] + rad)
                       end_angle = start_angle[1:] + [math.radians(0)]
                       colors = Greens[len(out)]
                       for i in range(len(out)):
                            graph.wedge(
        x=0, y=0, radius=0.8,
        start_angle=start_angle[i],
        end_angle=end_angle[i],
        color=colors[i],
        legend_label=f"{out[i]} - {round(data[i])}%"
    )

                       graph.add_layout(graph.legend[0], 'right')
                       st.bokeh_chart(graph, use_container_width=True)
                       st.header('More information on the subjects')
                       for index in l:
                           st.subheader(Graduate[index])
                           st.write(df['About'][index - 1])
                       st.header('Choice of Degrees')
                       for index in l:
                           st.subheader(Graduate[index])
                           st.write(df['avgsal'][index - 1])
                       def Convert(string):
                            li = list(string.split(","))
                            li = list(map(float, li))
                            return li
                       st.header('Trends over the years')
                       l = [1, 2, 3, 4, 5]
                       x = ['2000', '2005', '2010', '2015', '2020']
                       y = []
                       for i in range(0, 5):
                            t = Convert(df['trends'][int(l[i]) - 1])
                            y.append(t)
                       output_file("line.html")
                       graph2 = figure(title="Trends", x_axis_label='Year', y_axis_label='Value')

                       # Plot each line
                       colors = ["Purple", "Blue", "Green", "Magenta", "Red"]
                       for i in range(5):
                           graph2.line(x, y[i], line_color=colors[i], legend_label=out[i])

                       graph2.add_layout(graph2.legend[0], 'right')

                           # Show the plot in Streamlit
                       st.bokeh_chart(graph2, use_container_width=True)
        ######################################  UNDERGRADUATE #########################################
                       
            elif kr1 == "Undergraduate":
                lis = []
                questions = [
        "You can manage all aspects of information security and ensure the protection of a company's digital data.",
        "You enjoy analyzing a company's business and information needs and using that data to create processes that help meet strategic goals.",
        "You are skilled at identifying problems and either designing new systems or improving existing ones to enhance their efficiency.",
        "You excel at working with databases and large datasets, whether it's designing, developing, modifying, or editing them.",
        "You are adept at using BI software tools to analyze, compare, visualize, and clearly present data.",
        "You are skilled in setting up and supporting Microsoft's Dynamics CRM system.",
        "You can bring creativity and innovation to developing user-friendly mobile applications.",
        "You are effective in roles that blend psychology, business, market research, design, and technology.",
        "You are reliable in maintaining quality systems like laboratory controls, document management, and training to ensure smooth manufacturing processes.",
        "You are passionate about designing and developing websites, whether it's the front-end or back-end."]
                with st.form("Ubdergraduate_10_form"):
                     for i, question in enumerate(questions):
                         st.header(f"Question {i+1}")
                         st.write(question)
                         inp = st.selectbox( "", 
                    ["Select an Option", "Strongly Agree", "Agree", "Neutral", "Disagree", "Strongly Disagree"], 
                    key=str(i+1))
                         if inp != "Select an Option":
                             lis.append(qvals[inp])
                     submit_button = st.form_submit_button(label='Submit')
                if submit_button:
                   if len(lis) == 10:
                       st.success("Test Completed")
                       st.title("RESULTS:")
                       df = pd.read_csv(r"Occupations.csv")
                       input_list = lis
                       subjects = {
                             1: "Systems Security Administrator",
                2: "Business Systems Analyst",
                3: "Software Systems Engineer",
                4: "Database Developer",
                5: "Business Intelligence Analyst",
                6: "CRM Technical Developer",
                7: "Mobile Applications Developer",
                8: "UX Designer",
                9: "Quality Assurance Associate",
                10: "Web Developer"
                                      }
                       def output(listofanswers):
                            all_scores = [val / 5 for val in listofanswers]
                            sorted_scores = sorted(enumerate(all_scores, 1), key=lambda x: x[1], reverse=True)
                            top_indices = [x[0] for x in sorted_scores[:5]]
                            top_scores = [x[1] for x in sorted_scores[:5]]
                            normalized_scores = [x * 100 / sum(top_scores) for x in top_scores]
                            return top_indices, normalized_scores
                       l, data = output(input_list)
                       out = [subjects[i] for i in l]
                       output_file("pie.html")
                       graph = figure(title="Recommended subjects", height=500, width=500)
                       radians = [math.radians((percent / 100) * 360) for percent in data]
                       start_angle = [math.radians(0)]
                       for rad in radians[:-1]:
                           start_angle.append(start_angle[-1] + rad)
                       end_angle = start_angle[1:] + [math.radians(0)]
                       colors = Greens[len(out)]
                       for i in range(len(out)):
                            graph.wedge(
        x=0, y=0, radius=0.8,
        start_angle=start_angle[i],
        end_angle=end_angle[i],
        color=colors[i],
        legend_label=f"{out[i]} - {round(data[i])}%"
    )

                       graph.add_layout(graph.legend[0], 'right')
                       st.bokeh_chart(graph, use_container_width=True)
                       st.header('More information on the subjects')
                       for index in l:
                           st.subheader(subjects[index])
                           st.write(df['Information'][index - 1])
                       st.header('Choice of Degrees')
                       for index in l:
                           st.subheader(subjects[index])
                           st.write(df['Income'][index - 1])
                       def Convert(string):
                            li = list(string.split(","))
                            li = list(map(float, li))
                            return li
                       st.header('Trends over the years')
                       l = [1, 2, 3, 4, 5]
                       x = ['2000', '2005', '2010', '2015', '2020']
                       y = []
                       for i in range(0, 5):
                            t = Convert(df['trends'][int(l[i]) - 1])
                            y.append(t)
                       output_file("line.html")
                       graph2 = figure(title="Trends", x_axis_label='Year', y_axis_label='Value')

                       # Plot each line
                       colors = ["Purple", "Blue", "Green", "Magenta", "Red"]
                       for i in range(5):
                           graph2.line(x, y[i], line_color=colors[i], legend_label=out[i])

                       graph2.add_layout(graph2.legend[0], 'right')

                           # Show the plot in Streamlit
                       st.bokeh_chart(graph2, use_container_width=True)

       

if __name__=="__main__":
    main()
