import os
import requests
import gradio as gr
from dotenv import load_dotenv
from pydantic import BaseModel


from langchain_openai import ChatOpenAI
from langchain.tools import StructuredTool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub



load_dotenv("api.env")

print("OPENAI_API_KEY loaded:", bool(os.getenv("OPENAI_API_KEY")))
print("OPENWEATHER_API_KEY loaded:", bool(os.getenv("OPENWEATHER_API_KEY")))
print("CRICAPI_KEY loaded:", bool(os.getenv("CRICAPI_KEY")))
print("FOREX_API_KEY loaded:", bool(os.getenv("FOREX_API_KEY")))
print("TOMTOM_API_KEY loaded:", bool(os.getenv("TOMTOM_API_KEY")))
print("NEWSDATA_API_KEY loaded:", bool(os.getenv("NEWSDATA_API_KEY")))


def get_weather(city: str) -> str:
    api_key = os.getenv("OPENWEATHER_API_KEY")

    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={api_key}&units=metric"
    )

    data = requests.get(url).json()

    if data.get("cod") != 200:
        return f"❌ Could not fetch weather for {city}."

    temp = data["main"]["temp"]
    temp_min = data["main"]["temp_min"]
    temp_max = data["main"]["temp_max"]
    humidity = data["main"]["humidity"]
    condition = data["weather"][0]["description"]

    return (
        f"🌍 Weather in {city}\n"
        f"🌡 Current: {temp}°C\n"
        f"🔻 Low: {temp_min}°C\n"
        f"🔺 High: {temp_max}°C\n"
        f"☁ Condition: {condition}\n"
        f"💧 Humidity: {humidity}%"
    )

# ===== LIVE CRICKET SCORES =====
def get_live_cricket_scores() -> str:
    api_key = os.getenv("CRICAPI_KEY")
    url = f"https://api.cricapi.com/v1/currentMatches?apikey={api_key}&offset=0"
    data = requests.get(url, timeout=5).json()

    if not data.get("status"):
        return "❌ Unable to fetch live cricket scores."

    matches = data.get("data", [])
    if not matches:
        return "ℹ️ No live matches right now."

    response = "🏏 LIVE CRICKET SCORES\n\n"
    for match in matches[:3]:
        teams = " vs ".join(match.get("teams", []))
        status = match.get("status", "Status unavailable")
        response += f"🏟 {teams}\n📢 {status}\n\n"

    return response


# ===== FOREX =====
def get_forex_rate(base_currency: str, target_currency: str) -> str:
    api_key = os.getenv("FOREX_API_KEY")
    url = f"https://v6.exchangerate-api.com/v6/{api_key}/pair/{base_currency}/{target_currency}"
    data = requests.get(url, timeout=5).json()

    if data.get("result") != "success":
        return "❌ Unable to fetch forex rate."

    rate = data["conversion_rate"]
    return f"💱 1 {base_currency.upper()} = {rate} {target_currency.upper()}"


CITY_TO_IATA = {
    "chennai": "MAA",
    "delhi": "DEL",
    "mumbai": "BOM",
    "bangalore": "BLR",
    "kolkata": "CCU",
    "hyderabad": "HYD",
    "cochin": "COK",
    "trivandrum": "TRV",
    "ahmedabad": "AMD",
    "pune": "PNQ"
}

def search_flights(dep_iata: str, arr_iata: str) -> str:

    dep_iata = CITY_TO_IATA.get(dep_iata.lower(), dep_iata.upper())
    arr_iata = CITY_TO_IATA.get(arr_iata.lower(), arr_iata.upper())


    api_key = os.getenv("AVIATIONSTACK_API_KEY")

    url = "http://api.aviationstack.com/v1/flights"
    params = {
        "access_key": api_key,
        "dep_iata": dep_iata,
        "arr_iata": arr_iata
    }

    data = requests.get(url, params=params, timeout=5).json()
    flights = data.get("data", [])

    if not flights:
        return "❌ No flights found for this route."

    response = "✈️ Flights found:\n\n"
    for f in flights[:5]:
        response += (
            f"{f['airline']['name']} "
            f"{f['flight']['iata']} — "
            f"{f['flight_status']}\n"
        )

    response += "\nTell me the airline or flight number for details."
    return response


def get_flight_status(flight_number: str) -> str:
    api_key = os.getenv("AVIATIONSTACK_API_KEY")

    url = "http://api.aviationstack.com/v1/flights"
    params = {
        "access_key": api_key,
        "flight_iata": flight_number.upper()
    }

    data = requests.get(url, params=params, timeout=5).json()

    flights = data.get("data", [])
    if not flights:
        return f"❌ No live data found for flight {flight_number.upper()}."

    flight = flights[0]

    airline = flight["airline"]["name"]
    flight_no = flight["flight"]["iata"]
    status = flight["flight_status"]

    dep = flight["departure"]
    arr = flight["arrival"]

    return (
        f"✈️ Flight Status\n\n"
        f"🛫 Airline: {airline}\n"
        f"🔢 Flight: {flight_no}\n"
        f"📢 Status: {status.upper()}\n\n"
        f"📍 Departure: {dep['airport']} ({dep['iata']})\n"
        f"🕒 Scheduled: {dep['scheduled']}\n\n"
        f"📍 Arrival: {arr['airport']} ({arr['iata']})\n"
        f"🕒 Scheduled: {arr['scheduled']}"
    )

# ===== TRAFFIC ROUTE =====
def get_coordinates(city: str):

    api_key = os.getenv("TOMTOM_API_KEY")

    url = f"https://api.tomtom.com/search/2/geocode/{city}.json"

    params = {
        "key": api_key
    }

    data = requests.get(url, params=params, timeout=5).json()

    results = data.get("results")

    if not results:
        return None

    lat = results[0]["position"]["lat"]
    lon = results[0]["position"]["lon"]

    return f"{lat},{lon}"

def get_travel_time(origin: str, destination: str):

    api_key = os.getenv("TOMTOM_API_KEY")

    origin_coord = get_coordinates(origin)
    dest_coord = get_coordinates(destination)

    if not origin_coord or not dest_coord:
        return "❌ Unable to find coordinates for the given locations."

    url = f"https://api.tomtom.com/routing/1/calculateRoute/{origin_coord}:{dest_coord}/json"

    params = {
        "traffic": "true",
        "key": api_key
    }

    data = requests.get(url, params=params, timeout=5).json()

    routes = data.get("routes")

    if not routes:
        return "❌ Unable to fetch travel data."

    summary = routes[0]["summary"]

    distance = summary["lengthInMeters"] / 1000
    duration = summary["travelTimeInSeconds"] / 60

    return (
        f"🚗 Travel Info\n\n"
        f"📍 From: {origin}\n"
        f"📍 To: {destination}\n"
        f"📏 Distance: {distance:.1f} km\n"
        f"⏱ Travel Time: {duration:.1f} minutes"
    )

# === NEWS DATA ===
def get_city_news(city: str) -> str:

    api_key = os.getenv("NEWSDATA_API_KEY")

    url = "https://newsdata.io/api/1/news"

    params = {
        "apikey": api_key,
        "q": city,
        "language": "en"
    }

    data = requests.get(url, params=params, timeout=5).json()

    articles = data.get("results", [])

    if not articles:
        return f"📰 No recent news found for {city}."

    response = f"📰 Latest News in {city}\n\n"

    for article in articles[:5]:
        title = article["title"]
        source = article["source_id"]

        response += f"• {title}\n({source})\n\n"

    return response

# ======iMAGE GENERATION=====
def generate_image(prompt: str) -> str:
    import urllib.parse

    encoded = urllib.parse.quote(prompt)

    image_url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&seed=42"

    return image_url



class FlightSearchInput(BaseModel):
    dep_iata: str
    arr_iata: str

class ForexInput(BaseModel):
    base_currency: str
    target_currency: str


class WeatherInput(BaseModel):
    city: str

class FlightInput(BaseModel):
    flight_number: str

class ImageInput(BaseModel):
    prompt: str

class TravelInput(BaseModel):
    origin: str
    destination: str

class NewsInput(BaseModel):
    city:str

weather_tool = StructuredTool.from_function(
    func=get_weather,
    name="current_weather",
    description="Get current temperature, condition, and humidity for a city",
    
)

live_cricket_tool = StructuredTool.from_function(
    func=get_live_cricket_scores,
    name="live_cricket_scores",
    description="Get current live cricket match scores and status"
)

forex_tool = StructuredTool.from_function(
    func=get_forex_rate,
    name="forex_rate",
    description="Get live forex exchange rate between two currencies",
    args_schema=ForexInput
)

flight_search_tool = StructuredTool.from_function(
    func=search_flights,
    name="search_flights",
    description="Search flights by route using departure and arrival airport IATA codes",
    args_schema=FlightSearchInput
)


flight_status_tool = StructuredTool.from_function(
    func=get_flight_status,
    name="flight_status",
    description="Get live flight status using flight number (example: AI302, 6E203)",
    args_schema=FlightInput
)

travel_tool = StructuredTool.from_function(
    func=get_travel_time,
    name="travel_time",
    description="Get travel time and distance between two cities with traffic",
    args_schema=TravelInput
)

news_tool = StructuredTool.from_function(
    func=get_city_news,
    name="city_news",
    description="Get latest news headlines for a specific city",
    args_schema=NewsInput
)

llm = ChatOpenAI(
    model="google/gemini-2.0-flash-001",
    temperature=1,
    openai_api_base="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY"), 
)

image_tool = StructuredTool.from_function(
    func=generate_image,
    name="generate_image",
    description="Generate an image from a text description",
    args_schema=ImageInput
)


base_prompt = hub.pull("hwchase17/openai-tools-agent")

custom_rules = """
You are a live data fetching assistant calls tool when needed.

Instructions:
- If the user asks about weather, extract the city name.
- If the city name appears misspelled, ambiguous, or uncommon:
  - Do NOT call the weather tool.
  - Ask the user: "Did you mean <correct city>?"
- Wait for user confirmation ("yes", "correct", or city name repeated).
- Only after confirmation, call the weather tool.
- If the city is clear and common, call the weather tool immediately.
- If the user asks about weather → use weather tool
- If the user asks about live cricket scores or ongoing matches → use live_cricket_scores
- If the user asks about forex or currency rates → use forex_rate
- If the user asks about flight info without a flight number:
  - Extract source and destination cities if mentioned.
  - Call the flight search tool.
  - If multiple flights are found, ask the user to choose one.
  - When searching flights by route:
  - Use city names to infer airport IATA codes.
  - If source or destination is missing, ask one clarifying question.
- If the user asks about flight status or flight information:
  - Extract the flight number (example: AI302, 6E203).
  - Call the flight_status tool.
- If the user asks about travel time, traffic, distance, or route duration:
  - Extract origin and destination cities
  - Call the travel_time tool
  - If origin or destination is missing, ask the user
- If the user asks about news, headlines, or current events:
  - Extract the city name if mentioned
  - Call the city_news tool
  - If no city is provided, ask the user which city they want news for

- Otherwise answer normally

"""

prompt = ChatPromptTemplate.from_messages([
    ("system", custom_rules),
    *base_prompt.messages
])


agent = create_openai_tools_agent(
    llm=llm,
    tools=[
        weather_tool,
        live_cricket_tool,
        forex_tool,
        flight_search_tool,
        flight_status_tool,
        travel_tool,
        news_tool,
        image_tool
    ],
    prompt=prompt
)



memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)


agent_executor = AgentExecutor(
    agent=agent,
    tools=[
        weather_tool,
        live_cricket_tool,
        forex_tool,
        flight_search_tool,
        flight_status_tool,
        travel_tool,
        news_tool,
        image_tool
    ],
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3
)

def chat_handler(message, history):
    try:
        response = agent_executor.invoke({"input": message})
        output = response["output"]

        if "image.pollinations.ai" in output:
            return f"🎨 Generated Image:\n\n![image]({output})"

        return output

    except Exception:
        return "⚠️ Sorry, I couldn't retrieve the data right now."


demo = gr.ChatInterface(
    fn=chat_handler,
    title=" Chatbot",
    description="",
)

if __name__ == "__main__":
    demo.launch()
        




