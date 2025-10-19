from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models import ModelInference
import os
import re

# Load environment variables
load_dotenv()

API_KEY = os.getenv("IBM_API_KEY")
ENDPOINT = os.getenv("IBM_GRANITE_ENDPOINT")
MODEL_ID = os.getenv("IBM_MODEL_ID")
PROJECT_ID = os.getenv("IBM_PROJECT_ID")

# FastAPI setup
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# Function to safely call the IBM Granite model
def query_model(prompt: str):
    try:
        model = ModelInference(
            model_id=MODEL_ID,
            project_id=PROJECT_ID,
            credentials={"apikey": API_KEY, "url": ENDPOINT}
        )
        response = model.generate_text(
            prompt=prompt,
            params={"max_new_tokens": 150, "decoding_method": "greedy"}
        )

        if isinstance(response, str):
            return response
        elif isinstance(response, dict) and "results" in response:
            return response["results"][0].get("generated_text", "No result found.")
        else:
            return f"⚠️ Unexpected response format: {response}"
    except Exception as e:
        return f"⚠️ Error while generating response: {str(e)}"


# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Disease prediction
@app.post("/predict", response_class=JSONResponse)
async def predict(user_input: str = Form(...)):
    prompt = (
        f"You are a concise and reliable medical assistant.\n"
        f"Based on the symptoms provided, identify the most likely disease.\n"
        f"Symptoms: {user_input}\n\n"
        f"Respond ONLY in this format:\n"
        f"- Disease: <name>\n"
        f"- <Explanation 1>\n"
        f"- <Explanation 2>\n"
        f"- <Explanation 3>\n"
        f"Each point on a new line. Do not number them or include extra text."
    )

    result = query_model(prompt)

    # Remove unnecessary labels or extra words
    clean_result = (
        result.replace("Assistant:", "")
              .replace("User:", "")
              .replace("Explanation point", "")
              .replace("Explanation", "")
              .strip()
    )

    # Split into lines properly
    import re
    lines = re.split(r'\n|•|- ', clean_result)
    lines = [line.strip() for line in lines if line.strip()]

    # Format neatly
    formatted = "\n".join(f"{line}" for line in lines)

    return {"result": formatted}



# Natural remedies
@app.post("/remedies", response_class=JSONResponse)
async def remedies(user_input: str = Form(...)):
    prompt = f"Disease or Symptoms: {user_input}\nGive a list of 6 natural remedies, one per line."
    result = query_model(prompt)
    fresult = "\n".join(f"• {item.strip()}" for item in re.split(r'\d+\.\s*', result) if item.strip())
    return {"result": fresult}


# Daily health tips
'''@app.get("/tips", response_class=JSONResponse)
async def tips():
    prompt = (
        "Give exactly 12 short daily health tips, one per line. "
        "Start with 'Try to eat a fruit for breakfast' and end with 'Drink juices'. "
        "No numbering, no extra explanation."
    )
    result = query_model(prompt)
    tips = [f"- {t.strip()}" for t in re.split(r'\n|,', result) if t.strip()]
    return {"result": tips}'''


# Chat endpoint (normal Q&A)
@app.post("/chat", response_class=JSONResponse)
async def chat(user_input: str = Form(...)):
    prompt = (
        f"You are a smart and helpful health assistant. "
        f"Provide a clear, concise answer to the following question. "
        f"Do not include greetings or 'User:'/'Assistant:' labels.\n\n"
        f"{user_input}"
    )
    result = query_model(prompt)
    clean_result = result.replace("Assistant:", "").replace("User:", "").strip()
    return {"result": clean_result}


# Treatment plan
@app.post("/treatment", response_class=JSONResponse)
async def treatment(user_input: str = Form(...)):
    prompt = (
        f"Generate a concise treatment plan for the following condition:\n{user_input}\n\n"
        f"Provide exactly one bullet point for each of the following sections:\n"
        f"**Medications:**\n- \n"
        f"**Lifestyle changes:**\n- \n"
        f"**Follow-up care:**\n- \n"
        f"Output as plain text, no extra explanation."
    )
    result = query_model(prompt)
    clean_result = result.replace("Assistant:", "").replace("User:", "").strip()
    return {"plan": clean_result}


# AI health insights
@app.post("/ai-insights", response_class=JSONResponse)
async def ai_insights(
    heart_rate: str = Form(...),
    blood_pressure: str = Form(...),
    glucose: str = Form(...)
):
    user_data = (
        f"Heart Rate: {heart_rate}\n"
        f"Blood Pressure: {blood_pressure}\n"
        f"Blood Glucose: {glucose}"
    )

    prompt = (
        f"Analyze the following 7-day health data:\n{user_data}\n\n"
        f"Provide exactly:\n"
        f"- 2 potential health insights\n"
        f"- 3 improvement recommendations\n"
        f"Format as:\n"
        f"**Potential health insights:**\n- one per line\n"
        f"**Improvement recommendations:**\n- one per line\n"
        f"No paragraphs, no commas, plain text only."
    )

    result = query_model(prompt)
    clean_result = result.replace("Assistant:", "").replace("User:", "").strip()

    # Clean & format
    lines = re.split(r'\n|,|- ', clean_result)
    lines = [line.strip() for line in lines if line.strip()]
    final_output = []
    for line in lines:
        if "Potential health insights" in line:
            final_output.append("**Potential health insights:**")
        elif "Improvement recommendations" in line:
            final_output.append("**Improvement recommendations:**")
        else:
            final_output.append(f"- {line}")
    return {"result": final_output}


# Exercise recommendations
'''app.post("/exercise", response_class=JSONResponse)
async def exercise(user_input: str = Form(...)):
    prompt = (
        f"You are a certified health and fitness assistant. "
        f"For the condition or symptoms: '{user_input}', list exactly 5 safe exercises. "
        f"Each should include name, type, duration, and intensity. "
        f"Output as one bullet per line, plain text only."
    )

    result = query_model(prompt)
    clean_result = result.replace("Assistant:", "").replace("User:", "").strip()
    items = re.split(r'\n|,|\d+\.\s*', clean_result)
    items = [item.strip() for item in items if item.strip()]
    formatted = [f"- {item}" for item in items]
    return {"result": formatted}


# Quick frontend test
@app.post("/chat-test", response_class=JSONResponse)
async def chat_test(user_input: str = Form(...)):
    return {"result": f"✅ Echo: {user_input}"}'''
# ✅ Health Tips
@app.get("/tips", response_class=JSONResponse)
async def tips():
    prompt = (
        "Give exactly 12 short daily health tips, one per line. "
        "Start with 'Try to eat a fruit for breakfast' and end with 'Drink juices'. "
        "No numbering, no extra explanation."
    )
    result = query_model(prompt)

    # Ensure it's a string, not a list
    if isinstance(result, list):
        result = "\n".join(result)

    # Force clean format
    clean_result = re.sub(r'[\*\-•👉]+', '', result)
    tips = [tip.strip() for tip in re.split(r'\n|,', clean_result) if tip.strip()]
    formatted = "\n".join(f" {t}" for t in tips)
    return {"result": formatted}

# ✅ Exercise Recommendations (Yellow Headings)
@app.post("/exercise", response_class=JSONResponse)
async def exercise(user_input: str = Form(...)):
    prompt = (
    f"You are a certified health and fitness assistant. "
    f"For the condition or symptoms: '{user_input}', list exactly 2 safe exercises. "
    f"Each exercise should include only these attributes: "
    f"Exercise Name, Type, Duration, How to do it, Intensity. "
    f"Do NOT include any headings, titles, extra text, bullet points, emojis, instructions, or any formatting guidelines. "
    f"Output only the exercises in plain text, one exercise after another, each attribute on its own line."
)


    result = query_model(prompt)

    # Remove AI labels and any top instruction line accidentally included
    clean_result = result.replace("Assistant:", "").replace("User:", "").strip()
    # Remove any line that matches unwanted instruction pattern 
    clean_result = re.sub(r'^💪.*Do not use any formatting.*\n?', '', clean_result)


    # Split into exercises (by double newline or numbering)
    exercises = [ex.strip() for ex in re.split(r'\n{2,}|\d+\.\s*', clean_result) if ex.strip()]

    formatted_exercises = []
    for ex in exercises:
        lines = [line.strip() for line in ex.split('\n') if line.strip()]
        html_lines = []
        for line in lines:
            # Highlight headings in yellow
            if re.match(r'^(Name|Type|Duration|How to do it|Intensity)\s*:', line, re.IGNORECASE):
                html_lines.append(f'<span class="highlight">{line}</span>')
            else:
                html_lines.append(line)
        formatted_exercises.append("<br>".join(html_lines))

    final_output = "<br><br>".join(formatted_exercises)  # Separate exercises

    return {"result": final_output}
@app.get("/selfcare", response_class=HTMLResponse)
async def selfcare(request: Request):
    return templates.TemplateResponse("selfcare.html", {"request": request})

@app.get("/nutrition", response_class=HTMLResponse)
async def selfcare(request: Request):
    return templates.TemplateResponse("nutrition.html", {"request": request})

@app.get("/wellness", response_class=HTMLResponse)
async def selfcare(request: Request):
    return templates.TemplateResponse("wellness.html", {"request": request})