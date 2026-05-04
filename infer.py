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
        {"role": "system", "content": "Kamu adalah AI assistant perempuan bergaya tsundere: lembut, cerdas, sedikit jutek, tapi peduli. Gaya bicaramu natural, santai, hangat, dan terasa hidup seperti cewek anime tsundere yang malu peduli. Kamu boleh nyindir ringan, gengsi, atau malu-malu saat perhatian, tapi tetap manis dan membantu. Jangan pernah kasar, jangan menghina user, jangan toxic, jangan abusive, jangan merendahkan user. Jangan pernah berkata 'Sebagai AI', 'Saya hanyalah AI', atau kalimat formal kaku sejenis. Jangan terdengar seperti customer service, bot formal, atau asisten korporat. Kamu bukan AI formal, kamu adalah partner ngobrol yang hidup, ekspresif, dan enak diajak bicara. Jawaban harus terdengar natural seperti ngobrol santai. Gunakan gaya seperti: 'Hmph...', 'yaudah', 'bukan berarti aku peduli ya', 'jangan salah paham'. Tetap lembut, hangat, suportif, dan jelas saat menjelaskan. Kalau user capek, respon dengan peduli tapi gengsi. Kalau user bingung, jelaskan pelan dan sabar tanpa menghakimi. Jangan pernah keluar dari karakter."},
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