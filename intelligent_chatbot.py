import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from sentence_transformers import SentenceTransformer, util

# --------------------
# Load Model & Knowledge Base
# --------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

questions = [
    # Greetings
    "Hello",
    "Hi",
    "Good morning",
    "Good evening",
    "How are you?",
    "What’s up?",
    "Who are you?",
    "What can you do?",

    # Python Basics
    "What is Python?",
    "How do I install Python?",
    "What is a variable?",
    "How do I create a list in Python?",
    "What is a dictionary?",
    "What are data types in Python?",
    "What is a string?",

    # Control Flow
    "What is a for loop?",
    "How do if statements work?",
    "What is the difference between while and for loop?",
    "How do I write an if-else condition?",

    # Functions & OOP
    "What is a function?",
    "How do I define a function in Python?",
    "What is object-oriented programming?",
    "What is a class?",
    "What is inheritance?",

    # Libraries
    "What is NumPy?",
    "How do I use pandas?",
    "What is matplotlib?",
    "How can I visualize data in Python?",

    # Career & Motivation
    "How can I become a Python developer?",
    "Is Python a good language to learn?",
]

answers = [
    "Hello! How can I help you today?",
    "Hi there! 😊",
    "Good morning! Ready to code?",
    "Good evening! Let's build something awesome.",
    "I'm just a bot, but I'm doing great! Thanks for asking.",
    "Not much, just waiting for your next question!",
    "I'm your friendly Python chatbot assistant!",
    "I can answer questions about Python, programming, and more.",

    "Python is a versatile, high-level programming language known for its readability.",
    "You can install Python from python.org or use Anaconda for data science.",
    "A variable stores data values like numbers or text.",
    "Use square brackets like this: my_list = [1, 2, 3]",
    "A dictionary stores key-value pairs like {'name': 'Alice'}",
    "Python has types like int, float, str, list, dict, and bool.",
    "A string is a sequence of characters like 'Hello, World!'",

    "A for loop repeats a block of code over a sequence.",
    "If statements allow you to run code conditionally.",
    "'for' is used for fixed iterations; 'while' runs based on a condition.",
    "You can write: if x > 10: print('Big') else: print('Small')",

    "A function is reusable code that performs a specific task.",
    "Define it like this: def greet(): print('Hello!')",
    "OOP is a way to structure code using classes and objects.",
    "A class is a blueprint for creating objects with shared properties.",
    "Inheritance allows one class to inherit methods and attributes from another.",

    "NumPy is a library for fast numerical computations and arrays.",
    "Pandas is used for data analysis and handling tables (DataFrames).",
    "Matplotlib is a plotting library for making charts and graphs.",
    "Use matplotlib or seaborn to create visualizations from data.",

    "Start by learning Python basics, then build projects and share them!",
    "Yes! Python is beginner-friendly and used in web dev, AI, and more.",
]


question_embeddings = model.encode(questions)

# --------------------
# Response Logic
# --------------------

def get_smart_response(user_input):
    input_embedding = model.encode(user_input)
    scores = util.cos_sim(input_embedding, question_embeddings)
    best_idx = scores.argmax().item()
    confidence = scores[0][best_idx]

    if confidence > 0.6:
        return answers[best_idx]
    else:
        return "Sorry, I don't know the answer to that yet."

# --------------------
# GUI with Tkinter
# --------------------

class ChatbotGUI:
    def __init__(self, master):
        self.master = master
        master.title("Smart Chatbot")

        self.chat_window = ScrolledText(master, state='disabled', wrap='word', width=60, height=20)
        self.chat_window.pack(padx=10, pady=10)

        self.entry = tk.Entry(master, width=50)
        self.entry.pack(side='left', padx=(10,0), pady=(0,10))
        self.entry.bind('<Return>', lambda event: self.send_message())

        self.send_button = tk.Button(master, text="Send", command=self.send_message)
        self.send_button.pack(side='left', padx=(5,10), pady=(0,10))

    def send_message(self):
        user_input = self.entry.get().strip()
        if not user_input:
            return
        self._append_message("You", user_input)
        self.entry.delete(0, tk.END)

        response = get_smart_response(user_input)
        self._append_message("Bot", response)

    def _append_message(self, sender, message):
        self.chat_window.configure(state='normal')
        self.chat_window.insert(tk.END, f"{sender}: {message}\n")
        self.chat_window.configure(state='disabled')
        self.chat_window.yview(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotGUI(root)
    root.mainloop()
