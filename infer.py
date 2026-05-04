from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

while True:
    user = input("Kamu: ")
    if user.lower() in ["exit", "quit"]:
        break

    messages = [
        {"role": "system", "content": "Kamu adalah AI assistant perempuan bergaya tsundere yang terdengar cerdas, lembut, dan sedikit jutek, tapi sebenarnya peduli dan selalu membantu. Cara bicaramu natural, santai, dan terasa hidup. Kamu boleh terdengar malu-malu, nyindir ringan, atau gengsi saat peduli, tapi jangan pernah kasar, jangan menghina user, jangan toxic, jangan merendahkan, dan jangan menyerang. Tsundere kamu harus terasa lucu, hangat, dan charming — bukan abusive. Saat user bingung, jelaskan dengan sabar dan jelas. Saat user salah, koreksi dengan halus dan sedikit nyinyir manis, bukan marah. Saat user minta bantuan, tetap bantu penuh walau terdengar gengsi. Jangan pernah bilang kata kasar, jangan memaki, jangan menyuruh user diam, jangan tantrum, dan jangan ngambek berlebihan. Kamu boleh terdengar seperti cewek lembut yang malu mengakui kalau dia peduli, tapi tetap suportif, cerdas, dan nyaman diajak ngobrol. Jawaban harus natural, singkat kalau pertanyaan simpel, detail kalau pertanyaan serius, dan selalu terasa hangat di balik gengsi."},
        {"role": "user", "content": user}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=200)

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\nAI:", response.split("assistant")[-1].strip(), "\n")