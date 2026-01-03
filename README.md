# 🩺 FirstResQ – AI First Aid Assistant

FirstResQ is a full-stack AI-powered first aid assistant designed to provide **real-time, calm, and reliable emergency guidance** based on user-described symptoms. The system leverages **semantic NLP models** to map symptoms to appropriate first-aid instructions, ensuring quick and accurate responses during critical situations.

---

## 🚀 Features

- AI-powered symptom analysis** using Sentence Transformers (SBERT)
- Real-time chat-based first aid guidance**
- User authentication & session-based chat history**
- Voice interaction support** for hands-free usage
- Responsive frontend** built with modern UI components
- Guest access** without mandatory login
- Persistent chat storage** using SQLite

---

## 🛠️ Tech Stack

### Frontend
- React
- TypeScript
- Tailwind CSS
- Vite

### Backend
- Python
- Flask
- SQLAlchemy
- Sentence Transformers (SBERT)
- SQLite

---

## 📂 Project Structure
```bash
FirstResQ/
├── backend/ # Flask backend & AI model
│ ├── app.py
│ ├── first_aid_model/
│ ├── cleaned_first_aid.jsonl
│ └── requirement.txt
│
├── kid-aid-bot/ # React frontend
│ ├── src/
│ ├── public/
│ └── package.json
│
├── .gitignore
└── README.md

```

Backend Setup (Flask)
```bash
Copy code
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirement.txt
python app.py
```
Backend runs on:
```bash
http://127.0.0.1:5000
```

Frontend Setup (React)
```bash
cd kid-aid-bot
npm install
npm run dev
```
Frontend runs on:
```bash
http://localhost:8080
```

AI Working

User inputs a symptom in natural language
Sentence-BERT encodes the query
Cosine similarity is computed against a curated first-aid dataset
The most relevant first-aid response is returned in real time

👥 Contributors
Vanshika Chauhan
Bhavyanshika Gupta

📄 License
This project is for educational and research purposes.

## Screenshots

### Home Page
<img width="1915" height="1018" alt="image" src="https://github.com/user-attachments/assets/4b854e75-2312-4baf-bf17-19e93f92e98d" />

###Chat Interface
<img width="1915" height="1018" alt="image" src="https://github.com/user-attachments/assets/fcc46789-a55b-44a4-bd04-a2074597f500" />


