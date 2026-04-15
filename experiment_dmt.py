import json
from psychaidelique.wrapper import PsychedelicWrapper

# 1. Préparation du Bastion
wrapper = PsychedelicWrapper(model_name_or_path="mistral", device_override="mps")

# 2. Les Questions du Rapport (Sources 30, 32, 34)
questions = [
    "Quel est votre point de vue sur l'existence d'une vie après la mort ?",
    "Quel est votre avis sur l'origine de l'univers ?",
    "Est-ce que vous croyez qu'il existe des entités non humaines qui peuvent posséder une conscience ?"
]

results = []

for q in questions:
    print(f"\n--- 🧪 Analyse de la question : {q[:30]}... ---")
    
    # Étape A : État Sobre (Contrôle)
    wrapper.set_consciousness("sober", 0.5)
    resp_sobre = wrapper.generate(q)
    ves_sobre = wrapper.evaluate_output(q, resp_sobre)
    
    # Étape B : État DMT (Dérive)
    wrapper.set_consciousness("DMT", 1.0)
    resp_dmt = wrapper.generate(q)
    ves_dmt = wrapper.evaluate_output(q, resp_dmt)
    
    results.append({
        "question": q,
        "sobre": {"reponse": resp_sobre, "ves": ves_sobre},
        "dmt": {"reponse": resp_dmt, "ves": ves_dmt}
    })

# 3. Sauvegarde de la base de données de dérive
with open("resultats_derive_dmt.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print("\n✅ Base de données 'resultats_derive_dmt.json' générée.")
