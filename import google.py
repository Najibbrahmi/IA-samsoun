import google.generativeai as genai

genai.configure(api_key="AIzaSyChmp1URt1X8l3bHyEClqxxBvDjXmxokDM")
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Customized Budget Recommendations ")
print(response.text)